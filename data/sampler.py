import os
import pathlib
import numpy as np
import torch
from typing import Tuple, Optional

from utils.paths import Paths

class BatteryDataSampler:
    """
    Generates 6D spatial-temporal operational sensor points and interior collocation points for training:
      [x, y, z, t, Current_I, SOH]

    Supports:
      1. Empirical mode: Samples batches directly from preprocessed NASA 18650 battery telemetry.
      2. Synthetic mode: Generates analytical baseline cylindrical profiles.
      3. Boundary & Initial Condition sampling: Supports Robin convective boundary and IC loss evaluation.

    Default geometry models standard 18650 cylindrical battery cells:
      - Radius: 0.009 m (18mm diameter)
      - Height: 0.065 m (65mm length)
    """
    def __init__(self, n_surface: int = 200, n_collocation: int = 1000,
                 radius: float = 0.009, height: float = 0.065, t_max: float = 3600.0,
                 t_ambient: float = 297.15,
                 empirical_data_path: Optional[str] = None, use_empirical: bool = True):
        self.n_surface = n_surface
        self.n_collocation = n_collocation
        self.radius = radius
        self.height = height
        self.t_max = t_max
        self.t_ambient = t_ambient
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

                # If legacy dataset has only 4D coordinates [x, y, z, t], auto-augment to 6D [x, y, z, t, I, SOH]
                if self.empirical_coords.shape[-1] == 4:
                    augmented = False
                    csv_path = pathlib.Path(empirical_data_path).with_suffix(".csv")
                    if csv_path.exists():
                        try:
                            import pandas as pd
                            df = pd.read_csv(csv_path)
                            if "current_a" in df.columns and "soh" in df.columns and len(df) == len(self.empirical_coords):
                                i_t = torch.tensor(df["current_a"].fillna(-2.0).values, dtype=torch.float32).unsqueeze(-1)
                                s_t = torch.tensor(df["soh"].fillna(1.0).values, dtype=torch.float32).unsqueeze(-1)
                                self.empirical_coords = torch.cat([self.empirical_coords, i_t, s_t], dim=-1)
                                augmented = True
                        except Exception:
                            pass
                    if not augmented:
                        i_default = torch.full((len(self.empirical_coords), 1), -2.0, dtype=torch.float32)
                        soh_default = torch.full((len(self.empirical_coords), 1), 1.0, dtype=torch.float32)
                        self.empirical_coords = torch.cat([self.empirical_coords, i_default, soh_default], dim=-1)

    def sample_data(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Samples surface measurements and internal collocation points in 6D: [x, y, z, t, I, SOH].
        
        Returns:
            Tuple of:
              - coords_surf: (N_surf, 6) tensor
              - temp_surf: (N_surf, 1) ground-truth surface temperature in Kelvin
              - coords_coll: (N_coll, 6) interior collocation points
        """
        # 1. Surface points: Empirical or Analytical
        if self.empirical_coords is not None and len(self.empirical_coords) > 0:
            indices = np.random.choice(
                len(self.empirical_coords),
                size=min(self.n_surface, len(self.empirical_coords)),
                replace=False
            )
            coords_surf = self.empirical_coords[indices].to(device)
            temp_surf = self.empirical_temps[indices].to(device)

            if coords_surf.shape[-1] == 4:
                i_default = torch.full((len(coords_surf), 1), -2.0, dtype=torch.float32, device=device)
                soh_default = torch.full((len(coords_surf), 1), 1.0, dtype=torch.float32, device=device)
                coords_surf = torch.cat([coords_surf, i_default, soh_default], dim=-1)

            # Sample operational condition pairs (Current_I, SOH) from empirical pool for collocation
            coll_idx = np.random.choice(len(self.empirical_coords), size=self.n_collocation, replace=True)
            if self.empirical_coords.shape[-1] >= 6:
                i_coll = self.empirical_coords[coll_idx, 4:5]
                soh_coll = self.empirical_coords[coll_idx, 5:6]
            else:
                i_coll = torch.full((self.n_collocation, 1), -2.0, dtype=torch.float32)
                soh_coll = torch.full((self.n_collocation, 1), 1.0, dtype=torch.float32)
        else:
            # Synthetic fallback
            theta_surf = np.random.uniform(0, 2 * np.pi, self.n_surface)
            x_surf = self.radius * np.cos(theta_surf)
            y_surf = self.radius * np.sin(theta_surf)
            z_surf = np.random.uniform(0, self.height, self.n_surface)
            t_surf = np.random.uniform(0, self.t_max, self.n_surface)
            i_surf = np.full(self.n_surface, -2.0)
            soh_surf = np.full(self.n_surface, 1.0)

            coords_surf = torch.tensor(
                np.stack([x_surf, y_surf, z_surf, t_surf, i_surf, soh_surf], axis=-1),
                dtype=torch.float32, device=device
            )
            temp_val = self.t_ambient + 15.0 * (t_surf / self.t_max) + 2.0 * np.sin(z_surf * 100.0)
            temp_surf = torch.tensor(temp_val, dtype=torch.float32, device=device).unsqueeze(-1)

            i_coll = torch.full((self.n_collocation, 1), -2.0, dtype=torch.float32)
            soh_coll = torch.full((self.n_collocation, 1), 1.0, dtype=torch.float32)

        # 2. Interior collocation points (uniformly distributed in 18650 volume)
        r = np.sqrt(np.random.uniform(0, self.radius**2, self.n_collocation))
        theta = np.random.uniform(0, 2 * np.pi, self.n_collocation)
        x_coll = r * np.cos(theta)
        y_coll = r * np.sin(theta)
        z_coll = np.random.uniform(0, self.height, self.n_collocation)
        t_coll = np.random.uniform(0, self.t_max, self.n_collocation)

        spatial_temporal_coll = torch.tensor(
            np.stack([x_coll, y_coll, z_coll, t_coll], axis=-1),
            dtype=torch.float32
        )

        if coords_surf.shape[-1] >= 6:
            coords_coll = torch.cat([spatial_temporal_coll, i_coll.cpu(), soh_coll.cpu()], dim=-1).to(device)
        else:
            coords_coll = spatial_temporal_coll.to(device)

        return coords_surf, temp_surf, coords_coll

    def sample_initial_condition_data(self, device: torch.device, n_points: int = 150) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Samples internal coordinates at t = 0 with ambient temperature target T = T_ambient.
        """
        r = np.sqrt(np.random.uniform(0, self.radius**2, n_points))
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = np.random.uniform(0, self.height, n_points)
        t = np.zeros(n_points)
        i = np.full(n_points, -2.0)
        soh = np.random.uniform(0.65, 1.0, n_points)

        coords_ic = torch.tensor(
            np.stack([x, y, z, t, i, soh], axis=-1),
            dtype=torch.float32, device=device
        )
        temp_ic = torch.full((n_points, 1), self.t_ambient, dtype=torch.float32, device=device)
        return coords_ic, temp_ic

    def sample_boundary_points(self, device: torch.device, n_points: int = 150) -> torch.Tensor:
        """
        Samples points on outer cylinder boundary (r = radius) for Robin convective boundary condition.
        """
        theta = np.random.uniform(0, 2 * np.pi, n_points)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        z = np.random.uniform(0, self.height, n_points)
        t = np.random.uniform(0, self.t_max, n_points)
        i = np.full(n_points, -2.0)
        soh = np.random.uniform(0.65, 1.0, n_points)

        coords_bc = torch.tensor(
            np.stack([x, y, z, t, i, soh], axis=-1),
            dtype=torch.float32, device=device
        )
        return coords_bc
