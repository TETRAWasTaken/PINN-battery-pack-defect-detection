import os
import pathlib
import numpy as np
import torch
from typing import Tuple, Optional

from utils.paths import Paths

class BatteryDataSampler:
    """
    Generates surface sensor points and interior collocation points for training.
    Supports:
      1. Empirical mode: Samples batches directly from preprocessed NASA 18650 battery telemetry.
      2. Synthetic mode: Generates analytical baseline cylindrical profiles.

    Default geometry models standard 18650 cylindrical battery cells:
      - Radius: 0.009 m (18mm diameter)
      - Height: 0.065 m (65mm length)
    """
    def __init__(self, n_surface: int = 200, n_collocation: int = 1000,
                 radius: float = 0.009, height: float = 0.065, t_max: float = 3600.0,
                 empirical_data_path: Optional[str] = None, use_empirical: bool = True):
        self.n_surface = n_surface
        self.n_collocation = n_collocation
        self.radius = radius
        self.height = height
        self.t_max = t_max
        self.use_empirical = use_empirical

        self.empirical_coords = None
        self.empirical_temps = None

        if use_empirical:
            if empirical_data_path is None:
                default_pt = Paths.data_processed() / "pinn_surface_dataset.pt"
                if default_pt.exists():
                    empirical_data_path = str(default_pt)

            if empirical_data_path and os.path.exists(empirical_data_path):
                data = torch.load(empirical_data_path, map_location="cpu", weights_only=False)
                self.empirical_coords = data["coords"]
                self.empirical_temps = data["temp_k"]
                self.t_max = float(self.empirical_coords[:, 3].max().item())

    def sample_data(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples surface measurements and internal collocation points.
        
        Returns:
            Tuple of:
              - coords_surf: (N_surf, 4) tensor [x, y, z, t]
              - temp_surf: (N_surf, 1) ground-truth surface temperature in Kelvin
              - coords_coll: (N_coll, 4) interior collocation points [x, y, z, t]
        """
        # 1. Surface points: Empirical or Analytical
        if self.empirical_coords is not None and len(self.empirical_coords) > 0:
            indices = np.random.choice(len(self.empirical_coords), size=min(self.n_surface, len(self.empirical_coords)), replace=False)
            coords_surf = self.empirical_coords[indices].to(device)
            temp_surf = self.empirical_temps[indices].to(device)
        else:
            # Synthetic fallback
            theta_surf = np.random.uniform(0, 2 * np.pi, self.n_surface)
            x_surf = self.radius * np.cos(theta_surf)
            y_surf = self.radius * np.sin(theta_surf)
            z_surf = np.random.uniform(0, self.height, self.n_surface)
            t_surf = np.random.uniform(0, self.t_max, self.n_surface)

            coords_surf = torch.tensor(
                np.stack([x_surf, y_surf, z_surf, t_surf], axis=-1),
                dtype=torch.float32, device=device
            )
            # Synthetic 298.15 K base
            temp_val = 298.15 + 15.0 * (t_surf / self.t_max) + 2.0 * np.sin(z_surf * 100.0)
            temp_surf = torch.tensor(temp_val, dtype=torch.float32, device=device).unsqueeze(-1)

        # 2. Interior collocation points (always sampled across 18650 cylinder volume)
        r = np.sqrt(np.random.uniform(0, self.radius**2, self.n_collocation))
        theta = np.random.uniform(0, 2 * np.pi, self.n_collocation)
        x_coll = r * np.cos(theta)
        y_coll = r * np.sin(theta)
        z_coll = np.random.uniform(0, self.height, self.n_collocation)
        t_coll = np.random.uniform(0, self.t_max, self.n_collocation)

        coords_coll = torch.tensor(
            np.stack([x_coll, y_coll, z_coll, t_coll], axis=-1),
            dtype=torch.float32, device=device
        )

        return coords_surf, temp_surf, coords_coll
