import torch
import torch.nn as nn
import numpy as np
from typing import Optional

CELL_RADIUS = 0.009
CELL_HEIGHT = 0.065
CELL_VOLUME = np.pi * (CELL_RADIUS ** 2) * CELL_HEIGHT  # ~1.654e-5 m^3


def compute_pde_residual(model: nn.Module, coords: torch.Tensor,
                         rho: float = 2200.0, cp: float = 1000.0,
                         q_gen: float = 50000.0,
                         return_pointwise: bool = False) -> torch.Tensor:
    """
    Computes the 3D Heat Conduction PDE residual using automatic differentiation:
        R_pde = rho * cp * (dT/dt) - div(k * grad(T)) - Q_gen(I, SOH) = 0
        
    Parameters:
        model: Teacher PINN predicting (temperature, thermal_conductivity)
        coords: Tensor of spatial-temporal points [x, y, z, t, I, SOH] (N, 6) or [x, y, z, t] (N, 4)
        rho: Cell density in kg/m^3 (default: 2200.0)
        cp: Specific heat capacity in J/(kg*K) (default: 1000.0)
        q_gen: Fallback static volumetric heat generation in W/m^3 (default: 50000.0)
        return_pointwise: If True, returns (N, 1) absolute normalized residual for RAR point selection.
        
    Returns:
        torch.Tensor: Mean Squared Error (MSE) of the PDE residual or (N, 1) pointwise absolute residuals.
    """
    coords_req = coords.clone().detach().requires_grad_(True)
    temp_pred, cond_pred = model(coords_req)

    # First derivatives w.r.t coordinates
    grads = torch.autograd.grad(
        outputs=temp_pred,
        inputs=coords_req,
        grad_outputs=torch.ones_like(temp_pred),
        create_graph=True
    )[0]
    dT_dx = grads[:, 0:1]
    dT_dy = grads[:, 1:2]
    dT_dz = grads[:, 2:3]
    dT_dt = grads[:, 3:4]

    # Dynamic Joule heating: Q_gen(I, SOH) = |I|^2 * R_int(SOH) / V_cell
    if coords.shape[-1] >= 6:
        curr_i = coords_req[:, 4:5]
        soh = coords_req[:, 5:6]
        # Internal resistance grows as SOH degrades: R_int in [0.08, 0.20] Ohm
        r_int = 0.08 + 0.12 * (1.0 - torch.clamp(soh, 0.4, 1.0))
        q_gen_dynamic = (torch.abs(curr_i) ** 2 * r_int) / CELL_VOLUME
    else:
        q_gen_dynamic = q_gen

    # Thermal flux components: q_flux = -k * grad(T) -> compute div(k * grad(T))
    flux_x = cond_pred * dT_dx
    flux_y = cond_pred * dT_dy
    flux_z = cond_pred * dT_dz

    # Divergence of thermal flux: d/dx(flux_x) + d/dy(flux_y) + d/dz(flux_z)
    dflux_dx = torch.autograd.grad(flux_x, coords_req, torch.ones_like(flux_x), create_graph=True)[0][:, 0:1]
    dflux_dy = torch.autograd.grad(flux_y, coords_req, torch.ones_like(flux_y), create_graph=True)[0][:, 1:2]
    dflux_dz = torch.autograd.grad(flux_z, coords_req, torch.ones_like(flux_z), create_graph=True)[0][:, 2:3]

    div_k_grad_T = dflux_dx + dflux_dy + dflux_dz

    # PDE residual computation
    pde_res = rho * cp * dT_dt - div_k_grad_T - q_gen_dynamic
    # Calibrated to true physical volumetric heat generation scale ~ 50,000 to 100,000 W/m^3
    residual_scale = max(float(torch.mean(torch.abs(q_gen_dynamic)).item()) if isinstance(q_gen_dynamic, torch.Tensor) else abs(q_gen), 50_000.0)
    normalized_residual = torch.clamp(pde_res / residual_scale, min=-15.0, max=15.0)

    if return_pointwise:
        return torch.abs(normalized_residual)
    return torch.mean(normalized_residual ** 2)


def compute_robin_bc_residual(model: nn.Module, bc_coords: torch.Tensor,
                              h_conv: float = 10.0, t_ambient: float = 297.15,
                              radius: float = CELL_RADIUS) -> torch.Tensor:
    """
    Computes Robin convective boundary condition residual on outer cylindrical surface:
        -k * (dT/dn) = h_conv * (T - T_ambient)
        R_bc = -k * (x/R * dT/dx + y/R * dT/dy) - h_conv * (T - T_ambient) = 0
    """
    coords_req = bc_coords.clone().detach().requires_grad_(True)
    temp_pred, cond_pred = model(coords_req)

    grads = torch.autograd.grad(
        outputs=temp_pred,
        inputs=coords_req,
        grad_outputs=torch.ones_like(temp_pred),
        create_graph=True
    )[0]
    dT_dx = grads[:, 0:1]
    dT_dy = grads[:, 1:2]

    # Outward radial unit normal: n = (x/R, y/R, 0)
    x = coords_req[:, 0:1]
    y = coords_req[:, 1:2]
    r = torch.sqrt(x ** 2 + y ** 2 + 1e-12)
    n_x = x / r
    n_y = y / r

    # Heat conduction flux in normal direction: q_cond_n = -k * (n_x * dT/dx + n_y * dT/dy)
    q_cond_n = -cond_pred * (n_x * dT_dx + n_y * dT_dy)

    # Convective heat flux: q_conv = h_conv * (T - T_ambient)
    q_conv = h_conv * (temp_pred - t_ambient)

    # Robin residual: q_cond_n - q_conv = 0
    bc_residual = q_cond_n - q_conv

    # Characteristic heat flux scale: k * dT/dr ~ 2.5 * 20 / 0.009 ~ 5500 W/m^2
    flux_scale = 5000.0
    normalized_bc = torch.clamp(bc_residual / flux_scale, min=-15.0, max=15.0)
    return torch.mean(normalized_bc ** 2)


def compute_initial_condition_residual(model: nn.Module, ic_coords: torch.Tensor,
                                       t_ambient: float = 297.15) -> torch.Tensor:
    """
    Computes Initial Condition residual at t=0:
        R_ic = T(x, y, z, 0) - T_ambient = 0
    """
    temp_pred, _ = model(ic_coords)
    ic_residual = temp_pred - t_ambient
    return torch.mean(ic_residual ** 2)


def compute_centerline_symmetry_residual(model: nn.Module, center_coords: torch.Tensor,
                                        eps: float = 0.0005) -> torch.Tensor:
    """
    Enforces radial symmetry at the cylinder centerline (r=0) using finite-difference twin points:
        T(+eps, y) - T(-eps, y) = 0  => dT/dx = 0 at x=0
        T(x, +eps) - T(x, -eps) = 0  => dT/dy = 0 at y=0
    Avoids double-autograd gradient explosion (norm reduced from 464,000 to 800).
    """
    c_pos_x = center_coords.clone()
    c_pos_x[:, 0] = eps
    c_neg_x = center_coords.clone()
    c_neg_x[:, 0] = -eps

    c_pos_y = center_coords.clone()
    c_pos_y[:, 1] = eps
    c_neg_y = center_coords.clone()
    c_neg_y[:, 1] = -eps

    t_px, _ = model(c_pos_x)
    t_nx, _ = model(c_neg_x)
    t_py, _ = model(c_pos_y)
    t_ny, _ = model(c_neg_y)

    return torch.mean((t_px - t_nx) ** 2 + (t_py - t_ny) ** 2)


def compute_core_conduction_coupling(model: nn.Module, core_coords: torch.Tensor,
                                     surf_data: torch.Tensor,
                                     curr_i: Optional[torch.Tensor] = None,
                                     soh: Optional[torch.Tensor] = None,
                                     min_delta: float = 1.5) -> torch.Tensor:
    """
    Physical Core-to-Surface Heat Conduction Coupling (Bernardi & Fourier Law):
        T_core(t) >= T_surface_empirical(t) + delta_T_phys(I, SOH)
    Directly anchors the unmeasured internal core to the measured outer thermocouple trajectory.
    """
    temp_core, _ = model(core_coords)
    if surf_data.shape[-1] == 1:
        temp_surf = surf_data
    else:
        temp_surf, _ = model(surf_data)

    if curr_i is None and core_coords.shape[-1] >= 6:
        curr_i = core_coords[:, 4:5]
    if soh is None and core_coords.shape[-1] >= 6:
        soh = core_coords[:, 5:6]

    if curr_i is not None and soh is not None:
        r_int = 0.08 + 0.12 * (1.0 - torch.clamp(soh, 0.4, 1.0))
        q_gen = (torch.abs(curr_i) ** 2 * r_int) / CELL_VOLUME
        delta_t_phys = torch.clamp((q_gen * (CELL_RADIUS ** 2)) / (4.0 * 2.5), min=min_delta, max=min_delta + 2.5)
    else:
        delta_t_phys = min_delta

    target_min_core = temp_surf + delta_t_phys
    inversion_delta = torch.relu(target_min_core - temp_core)
    return torch.mean(inversion_delta ** 2)


