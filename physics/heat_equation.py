import torch
import torch.nn as nn

def compute_pde_residual(model: nn.Module, coords: torch.Tensor,
                         rho: float = 2200.0, cp: float = 1000.0, q_gen: float = 50000.0) -> torch.Tensor:
    """
    Computes the 3D Heat Conduction PDE residual using automatic differentiation:
        R_pde = rho * cp * (dT/dt) - div(k * grad(T)) - Q_gen = 0
        
    Parameters:
        model: Teacher PINN predicting (temperature, thermal_conductivity)
        coords: Tensor of spatial-temporal points [x, y, z, t] of shape (N, 4)
        rho: Cell density in kg/m^3 (default: 2200.0)
        cp: Specific heat capacity in J/(kg*K) (default: 1000.0)
        q_gen: Internal volumetric heat generation rate in W/m^3 (default: 50000.0)
        
    Returns:
        torch.Tensor: Mean Squared Error (MSE) of the PDE residual across all collocation points.
    """
    coords_req = coords.clone().detach().requires_grad_(True)
    temp_pred, cond_pred = model(coords_req)

    # First derivatives w.r.t coordinates [x, y, z, t]
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

    # Thermal flux components: q_flux = -k * grad(T) -> we compute div(k * grad(T))
    flux_x = cond_pred * dT_dx
    flux_y = cond_pred * dT_dy
    flux_z = cond_pred * dT_dz

    # Divergence of thermal flux: d/dx(flux_x) + d/dy(flux_y) + d/dz(flux_z)
    dflux_dx = torch.autograd.grad(flux_x, coords_req, torch.ones_like(flux_x), create_graph=True)[0][:, 0:1]
    dflux_dy = torch.autograd.grad(flux_y, coords_req, torch.ones_like(flux_y), create_graph=True)[0][:, 1:2]
    dflux_dz = torch.autograd.grad(flux_z, coords_req, torch.ones_like(flux_z), create_graph=True)[0][:, 2:3]

    div_k_grad_T = dflux_dx + dflux_dy + dflux_dz

    # PDE residual computation
    pde_res = rho * cp * dT_dt - div_k_grad_T - q_gen
    # Normalize by the source scale to keep the optimization objective numerically stable.
    residual_scale = max(abs(q_gen), rho * cp * 50.0 / 300.0, 1_000_000.0)
    normalized_residual = torch.clamp(pde_res / residual_scale, min=-100.0, max=100.0)
    return torch.mean(normalized_residual ** 2)
