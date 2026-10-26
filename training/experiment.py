import os
import json
import time
from typing import Dict, Any, Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.model_config import ModelConfig, set_seed
from data.sampler import BatteryDataSampler
from models.teacher_pinn import TeacherPINN
from models.student_mlp import StudentMLP
from physics.heat_equation import (
    compute_pde_residual,
    compute_robin_bc_residual,
    compute_initial_condition_residual,
    compute_centerline_symmetry_residual,
    compute_core_conduction_coupling
)


def _evaluate(
    teacher: nn.Module,
    student: nn.Module,
    sampler: BatteryDataSampler,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate both models on a fresh, deterministic holdout batch."""
    teacher.eval()
    student.eval()
    with torch.enable_grad():
        coords_surf, temp_surf, coords_coll = sampler.sample_data(device)
        teacher_temp, teacher_cond = teacher(coords_surf)
        teacher_surface_mse = nn.functional.mse_loss(teacher_temp, temp_surf)
        teacher_pde_mse = compute_pde_residual(teacher, coords_coll)

        coords_ic, _ = sampler.sample_initial_condition_data(device, n_points=100)
        coords_bc = sampler.sample_boundary_points(device, n_points=100)
        coords_center = sampler.sample_centerline_points(device, n_points=100)
        coords_core = coords_surf.clone()
        coords_core[:, 0] = 0.0
        coords_core[:, 1] = 0.0

        teacher_ic_mse = compute_initial_condition_residual(teacher, coords_ic)
        teacher_bc_mse = compute_robin_bc_residual(teacher, coords_bc)
        teacher_center_mse = compute_centerline_symmetry_residual(teacher, coords_center)
        teacher_core_mse = compute_core_conduction_coupling(teacher, coords_core, temp_surf)

    eval_coords = torch.cat([coords_surf, coords_coll], dim=0)
    with torch.no_grad():
        teacher_targets = torch.cat(teacher(eval_coords), dim=-1)
        student_preds = student(eval_coords)
        student_mse = nn.functional.mse_loss(student_preds, teacher_targets)
        student_temp_mse = nn.functional.mse_loss(student_preds[:, :1], teacher_targets[:, :1])
        student_cond_mse = nn.functional.mse_loss(student_preds[:, 1:], teacher_targets[:, 1:])

    return {
        "teacher_surface_mse": float(teacher_surface_mse.item()),
        "teacher_pde_mse": float(teacher_pde_mse.item()),
        "teacher_ic_mse": float(teacher_ic_mse.item()),
        "teacher_bc_mse": float(teacher_bc_mse.item()),
        "teacher_center_mse": float(teacher_center_mse.item()),
        "teacher_core_mse": float(teacher_core_mse.item()),
        "student_mse": float(student_mse.item()),
        "student_temp_mse": float(student_temp_mse.item()),
        "student_cond_mse": float(student_cond_mse.item()),
    }


def plot_experiment_convergence(
    teacher_history: Dict[str, List[float]],
    student_history: Dict[str, List[float]],
    config: ModelConfig,
    metrics: Dict[str, Any],
    save_path: str,
):
    """Generates and saves dual-panel convergence graphs for Teacher PINN and Student MLP."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    epochs_t = teacher_history["epoch"]
    total_loss = teacher_history["total_loss"]
    surf_loss = teacher_history["surf_loss"]
    pde_loss = teacher_history["pde_loss"]
    ic_loss = teacher_history.get("ic_loss", [])
    bc_loss = teacher_history.get("bc_loss", [])

    # --- Left Panel: Teacher PINN Convergence ---
    line1 = ax1.plot(epochs_t, total_loss, label="Total Loss", color="#1f77b4", linewidth=2)
    line2 = ax1.plot(epochs_t, surf_loss, label="Surface Sensor MSE", color="#2ca02c", linestyle="--", linewidth=1.8)
    lines_added = line1 + line2

    if ic_loss and len(ic_loss) == len(epochs_t):
        line_ic = ax1.plot(epochs_t, ic_loss, label="IC Residual", color="#ff7f0e", linestyle="-.", linewidth=1.5)
        lines_added += line_ic
    if bc_loss and len(bc_loss) == len(epochs_t):
        line_bc = ax1.plot(epochs_t, bc_loss, label="Robin BC Residual", color="#17becf", linestyle=":", linewidth=1.5)
        lines_added += line_bc

    ax1.set_xlabel("Teacher Epoch", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Loss / MSE", fontsize=11, color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Use twinx for PDE residual
    ax1_twin = ax1.twinx()
    line_pde = ax1_twin.plot(epochs_t, pde_loss, label="PDE Residual", color="#d62728", linestyle=":", linewidth=2)
    ax1_twin.set_ylabel("PDE Residual (MSE)", fontsize=11, color="#d62728")
    ax1_twin.tick_params(axis="y", labelcolor="#d62728")
    ax1_twin.set_yscale("log")

    all_lines = lines_added + line_pde
    labels = [l.get_label() for l in all_lines]
    ax1.legend(all_lines, labels, loc="upper right", framealpha=0.9)
    ax1.set_title(f"Teacher PINN Convergence (Final Surface MSE: {surf_loss[-1]:.4f})", fontsize=12, fontweight="bold")

    # --- Right Panel: Student MLP Distillation ---
    epochs_s = student_history["epoch"]
    distill_loss = student_history["distill_loss"]
    ax2.plot(epochs_s, distill_loss, label="Distillation Loss (MSE)", color="#9467bd", linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Student Epoch", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Distillation Loss (MSE)", fontsize=11, color="#9467bd")
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_title(f"Student MLP Distillation (Final Loss: {distill_loss[-1]:.6f})", fontsize=12, fontweight="bold")

    # Overall Figure Header
    fig.suptitle(
        f"Experiment: {config.exp_id} | Teacher Final Surf MSE: {surf_loss[-1]:.4f} | "
        f"Student Final MSE: {distill_loss[-1]:.6f}",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def run_single_experiment(
    config: ModelConfig,
    sampler: BatteryDataSampler,
    device: torch.device,
    output_dir: str = "experiments",
    save_plots: bool = True
) -> Dict[str, Any]:
    """
    Executes a single experiment configuration integrating all 5 advanced techniques:
      1. Conditional PINN (cPINN): 6D coordinates [x, y, z, t, I, SOH].
      2. Wang et al. Multiplicative Gated PINN architecture.
      3. Explicit Initial Condition and Robin Convective Boundary Condition.
      4. Two-Stage Hybrid Optimization (Adam exploration -> L-BFGS polishing).
      5. Residual-Based Adaptive Refinement (RAR) for collocation densification.
    """
    print(f"\n==================================================")
    print(f" Running Experiment: {config.exp_id}")
    print(f" Config: LR_T={config.lr_teacher:.2e}, Hidden={config.teacher_hidden_dim}, "
          f"FourierScale={config.fourier_scale}, GatedBackbone={config.use_gated_backbone}, "
          f"LambdaPDE={config.lambda_pde}, LambdaIC={config.lambda_ic}, LambdaBC={config.lambda_bc}")
    print(f"==================================================")

    start_time = time.time()
    set_seed(config.seed)

    # 1. Instantiate 6D Teacher PINN with Wang Gated Backbone
    teacher = TeacherPINN(
        in_dim=6,
        hidden_dim=config.teacher_hidden_dim,
        scale=config.fourier_scale,
        scale_space=config.scale_space,
        scale_time=config.scale_time,
        scale_cond=config.scale_cond,
        coordinate_scale=(0.009, 0.009, 0.065, 3600.0, 2.0, 1.0),
        t_ambient=297.15,
        use_gated_backbone=config.use_gated_backbone
    ).to(device)

    optimizer_teacher = optim.Adam(teacher.parameters(), lr=config.lr_teacher, weight_decay=1e-5)
    scheduler_teacher = None
    if config.use_lr_scheduler:
        scheduler_teacher = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_teacher, T_max=config.epochs_teacher, eta_min=1e-5
        )

    teacher_history = {
        "epoch": [],
        "total_loss": [],
        "surf_loss": [],
        "pde_loss": [],
        "ic_loss": [],
        "bc_loss": [],
        "center_loss": [],
        "core_loss": [],
        "lr": []
    }
    best_teacher_loss = float("inf")
    best_teacher_state = None

    # Adam Exploration Stage
    for epoch in range(1, config.epochs_teacher + 1):
        coords_surf, temp_surf, coords_coll = sampler.sample_data(device)
        coords_ic, _ = sampler.sample_initial_condition_data(device, n_points=min(150, config.batch_size // 2))
        coords_bc = sampler.sample_boundary_points(device, n_points=min(150, config.batch_size // 2))
        coords_center = sampler.sample_centerline_points(device, n_points=min(100, config.batch_size // 2))
        coords_core = coords_surf.clone()
        coords_core[:, 0] = 0.0
        coords_core[:, 1] = 0.0

        # Technique 5: Residual-Based Adaptive Refinement (RAR)
        if config.use_rar and (epoch % config.rar_frequency == 0):
            with torch.enable_grad():
                n_cand = config.rar_points_per_step * 5
                r_c = np.sqrt(np.random.uniform(0, 0.009**2, n_cand))
                th_c = np.random.uniform(0, 2 * np.pi, n_cand)
                x_c = r_c * np.cos(th_c)
                y_c = r_c * np.sin(th_c)
                z_c = np.random.uniform(0, 0.065, n_cand)
                t_c = np.random.uniform(0, 3600.0, n_cand)
                i_c = np.random.uniform(-2.5, 0.0, n_cand)
                soh_c = np.random.uniform(0.5, 1.0, n_cand)
                cand_coords = torch.tensor(
                    np.stack([x_c, y_c, z_c, t_c, i_c, soh_c], axis=-1),
                    dtype=torch.float32, device=device
                )
                pointwise_res = compute_pde_residual(teacher, cand_coords, return_pointwise=True)
                top_idx = torch.topk(pointwise_res.squeeze(-1), k=config.rar_points_per_step).indices
                refined_pts = cand_coords[top_idx].detach()
                coords_coll = torch.cat([coords_coll, refined_pts], dim=0)

        # 1. Surface Sensor Loss
        pred_surf_temp, _ = teacher(coords_surf)
        loss_surf = nn.functional.mse_loss(pred_surf_temp, temp_surf)

        # 2. Physics-informed PDE residual loss
        loss_pde = compute_pde_residual(teacher, coords_coll)

        # 3. Initial Condition loss
        loss_ic = compute_initial_condition_residual(teacher, coords_ic)

        # 4. Robin Boundary Condition loss
        loss_bc = compute_robin_bc_residual(teacher, coords_bc)

        # 5. Centerline Symmetry constraint (dT/dr = 0 at r=0)
        loss_center = compute_centerline_symmetry_residual(teacher, coords_center)

        # 6. Physical Core-to-Surface Thermal Ordering (T_core >= T_surface + Delta_T)
        loss_core = compute_core_conduction_coupling(teacher, coords_core, temp_surf)

        if config.use_adaptive_loss:
            optimizer_teacher.zero_grad()
            loss_surf.backward(retain_graph=True)
            grad_surf_norm = torch.cat([p.grad.flatten() for p in teacher.parameters() if p.grad is not None]).norm()

            optimizer_teacher.zero_grad()
            loss_pde.backward(retain_graph=True)
            grad_pde_norm = torch.cat([p.grad.flatten() for p in teacher.parameters() if p.grad is not None]).norm()

            adaptive_lambda = (grad_surf_norm / (grad_pde_norm + 1e-8)).clamp(min=1e-4, max=0.05).item()
            effective_lambda_pde = adaptive_lambda
        else:
            effective_lambda_pde = config.lambda_pde

        optimizer_teacher.zero_grad()
        total_loss = (
            config.lambda_surf * loss_surf
            + effective_lambda_pde * loss_pde
            + config.lambda_ic * loss_ic
            + config.lambda_bc * loss_bc
            + config.lambda_center * loss_center
            + config.lambda_core * loss_core
        )
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer_teacher.step()

        current_lr = optimizer_teacher.param_groups[0]["lr"]
        if scheduler_teacher is not None:
            scheduler_teacher.step()

        teacher_history["epoch"].append(epoch)
        teacher_history["total_loss"].append(total_loss.item())
        teacher_history["surf_loss"].append(loss_surf.item())
        teacher_history["pde_loss"].append(loss_pde.item())
        teacher_history["ic_loss"].append(loss_ic.item())
        teacher_history["bc_loss"].append(loss_bc.item())
        teacher_history["center_loss"].append(loss_center.item())
        teacher_history["core_loss"].append(loss_core.item())
        teacher_history["lr"].append(current_lr)

        if total_loss.item() < best_teacher_loss:
            best_teacher_loss = total_loss.item()
            best_teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}

        if epoch % max(1, config.epochs_teacher // 5) == 0 or epoch == 1:
            print(f" [Teacher Epoch {epoch:03d}/{config.epochs_teacher}] "
                  f"Total: {total_loss.item():.4f} | "
                  f"Surf: {loss_surf.item():.4f} | "
                  f"PDE: {loss_pde.item():.4f} | "
                  f"Core: {loss_core.item():.4f} | "
                  f"Symm: {loss_center.item():.4f} | LR: {current_lr:.2e}")

    # Technique 4: Stage 2 L-BFGS Second-Order Curvature Polishing
    if config.lbfgs_epochs > 0:
        print(f"\n -> Starting Stage 2: L-BFGS Second-Order Optimization ({config.lbfgs_epochs} epochs)...")
        optimizer_lbfgs = optim.LBFGS(
            teacher.parameters(),
            lr=0.5,
            max_iter=20,
            history_size=10,
            line_search_fn="strong_wolfe"
        )
        for l_epoch in range(1, config.lbfgs_epochs + 1):
            coords_surf, temp_surf, coords_coll = sampler.sample_data(device)
            coords_ic, _ = sampler.sample_initial_condition_data(device)
            coords_bc = sampler.sample_boundary_points(device)
            coords_center = sampler.sample_centerline_points(device)
            coords_core = coords_surf.clone()
            coords_core[:, 0] = 0.0
            coords_core[:, 1] = 0.0

            def closure():
                optimizer_lbfgs.zero_grad()
                pred_s, _ = teacher(coords_surf)
                ls = nn.functional.mse_loss(pred_s, temp_surf)
                lp = compute_pde_residual(teacher, coords_coll)
                lic = compute_initial_condition_residual(teacher, coords_ic)
                lbc = compute_robin_bc_residual(teacher, coords_bc)
                lcenter = compute_centerline_symmetry_residual(teacher, coords_center)
                lcore = compute_core_conduction_coupling(teacher, coords_core, temp_surf)
                step_loss = (
                    config.lambda_surf * ls
                    + config.lambda_pde * lp
                    + config.lambda_ic * lic
                    + config.lambda_bc * lbc
                    + config.lambda_center * lcenter
                    + config.lambda_core * lcore
                )
                step_loss.backward()
                return step_loss

            loss_val = optimizer_lbfgs.step(closure)
            pred_surf_now, _ = teacher(coords_surf)
            cur_surf_loss = nn.functional.mse_loss(pred_surf_now, temp_surf).item()
            if loss_val.item() < best_teacher_loss:
                best_teacher_loss = loss_val.item()
                best_teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}
            print(f" [L-BFGS Epoch {l_epoch:02d}/{config.lbfgs_epochs}] Loss: {loss_val.item():.4f} | Surf: {cur_surf_loss:.4f}")

    # Load best teacher model before starting Knowledge Distillation
    if best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)
    teacher.eval()

    # 2. Knowledge Distillation to Student MLP (6D inputs)
    print("\n -> Starting Knowledge Distillation to Student MLP...")
    student = StudentMLP(
        in_dim=6,
        hidden_dim=config.student_hidden_dim,
        out_dim=2,
        coordinate_scale=(0.009, 0.009, 0.065, 3600.0, 2.0, 1.0),
        t_ambient=297.15
    ).to(device)

    optimizer_student = optim.Adam(student.parameters(), lr=config.lr_student, weight_decay=1e-5)
    scheduler_student = None
    if config.use_lr_scheduler:
        scheduler_student = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_student, T_max=config.epochs_student, eta_min=1e-5
        )

    student_history = {
        "epoch": [],
        "distill_loss": []
    }
    best_student_loss = float("inf")
    best_student_state = None
    teacher.eval()

    steps_per_epoch = 10
    for epoch in range(1, config.epochs_student + 1):
        epoch_losses = []
        for _ in range(steps_per_epoch):
            coords_surf, _, coords_coll = sampler.sample_data(device)
            coords_core = coords_surf.clone()
            coords_core[:, 0] = 0.0
            coords_core[:, 1] = 0.0
            coords_center = sampler.sample_centerline_points(device, n_points=50)
            eval_coords = torch.cat([coords_surf, coords_coll, coords_core, coords_center], dim=0)

            with torch.no_grad():
                t_temp, t_cond = teacher(eval_coords)
                teacher_targets = torch.cat([t_temp, t_cond], dim=-1)

            optimizer_student.zero_grad()
            student_preds = student(eval_coords)
            distill_loss = nn.functional.mse_loss(student_preds, teacher_targets)
            distill_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer_student.step()
            epoch_losses.append(distill_loss.item())

        mean_loss = float(np.mean(epoch_losses))
        if scheduler_student is not None:
            scheduler_student.step()

        student_history["epoch"].append(epoch)
        student_history["distill_loss"].append(mean_loss)

        if mean_loss < best_student_loss:
            best_student_loss = mean_loss
            best_student_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        if epoch % max(1, config.epochs_student // 3) == 0 or epoch == 1:
            print(f" [Student Epoch {epoch:03d}/{config.epochs_student}] "
                  f"Distillation Loss: {mean_loss:.6f}")

    if best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)
    if best_student_state is not None:
        student.load_state_dict(best_student_state)

    # Deterministic evaluation on holdout batch
    set_seed(config.seed + 1)
    evaluation_metrics = _evaluate(teacher, student, sampler, device)
    elapsed_time = time.time() - start_time

    # 3. Save Experiment Artifacts
    exp_dir = os.path.join(output_dir, config.exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    torch.save(teacher.state_dict(), os.path.join(exp_dir, "teacher_pinn.pt"))
    torch.save(student.state_dict(), os.path.join(exp_dir, "student_mlp.pt"))
    torch.save({"model": teacher.state_dict(), "config": config.to_dict()},
               os.path.join(exp_dir, "teacher_checkpoint.pt"))
    torch.save({"model": student.state_dict(), "config": config.to_dict()},
               os.path.join(exp_dir, "student_checkpoint.pt"))

    metrics = {
        "exp_id": config.exp_id,
        "config": config.to_dict(),
        "final_teacher_loss": teacher_history["total_loss"][-1],
        "final_surface_loss": teacher_history["surf_loss"][-1],
        "final_pde_loss": teacher_history["pde_loss"][-1],
        "final_ic_loss": teacher_history["ic_loss"][-1],
        "final_bc_loss": teacher_history["bc_loss"][-1],
        "final_center_loss": teacher_history["center_loss"][-1] if teacher_history["center_loss"] else 0.0,
        "final_core_loss": teacher_history["core_loss"][-1] if teacher_history["core_loss"] else 0.0,
        "final_student_distill_loss": student_history["distill_loss"][-1],
        "best_teacher_surface_loss": best_teacher_loss,
        "best_student_distill_loss": best_student_loss,
        "evaluation": evaluation_metrics,
        "teacher_params": sum(p.numel() for p in teacher.parameters()),
        "student_params": sum(p.numel() for p in student.parameters()),
        "training_time_seconds": round(elapsed_time, 2)
    }

    # Save metrics JSON
    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    # Save training loss history CSV
    history_records = []
    max_epochs = max(len(teacher_history["epoch"]), len(student_history["epoch"]))
    for i in range(max_epochs):
        rec = {
            "epoch": i + 1,
            "teacher_total_loss": teacher_history["total_loss"][i] if i < len(teacher_history["epoch"]) else None,
            "teacher_surf_loss": teacher_history["surf_loss"][i] if i < len(teacher_history["epoch"]) else None,
            "teacher_pde_loss": teacher_history["pde_loss"][i] if i < len(teacher_history["epoch"]) else None,
            "teacher_ic_loss": teacher_history["ic_loss"][i] if i < len(teacher_history["ic_loss"]) else None,
            "teacher_bc_loss": teacher_history["bc_loss"][i] if i < len(teacher_history["bc_loss"]) else None,
            "teacher_center_loss": teacher_history["center_loss"][i] if i < len(teacher_history["center_loss"]) else None,
            "teacher_core_loss": teacher_history["core_loss"][i] if i < len(teacher_history["core_loss"]) else None,
            "teacher_lr": teacher_history["lr"][i] if i < len(teacher_history["lr"]) else None,
            "student_distill_loss": student_history["distill_loss"][i] if i < len(student_history["epoch"]) else None,
        }
        history_records.append(rec)
    pd.DataFrame(history_records).to_csv(os.path.join(exp_dir, "training_history.csv"), index=False)

    # Save flattened single-experiment metrics CSV
    flat_metrics = {
        "exp_id": config.exp_id,
        "lr_teacher": config.lr_teacher,
        "lr_student": config.lr_student,
        "teacher_hidden_dim": config.teacher_hidden_dim,
        "fourier_scale": config.fourier_scale,
        "scale_space": config.scale_space or config.fourier_scale,
        "scale_time": config.scale_time or config.fourier_scale,
        "lambda_pde": config.lambda_pde,
        "lambda_surf": config.lambda_surf,
        "lambda_ic": config.lambda_ic,
        "lambda_bc": config.lambda_bc,
        "lambda_center": config.lambda_center,
        "lambda_core": config.lambda_core,
        "use_gated_backbone": config.use_gated_backbone,
        "lbfgs_epochs": config.lbfgs_epochs,
        "use_rar": config.use_rar,
        "epochs_teacher": config.epochs_teacher,
        "epochs_student": config.epochs_student,
        "final_teacher_loss": metrics["final_teacher_loss"],
        "final_surface_loss": metrics["final_surface_loss"],
        "final_pde_loss": metrics["final_pde_loss"],
        "final_ic_loss": metrics["final_ic_loss"],
        "final_bc_loss": metrics["final_bc_loss"],
        "final_center_loss": metrics["final_center_loss"],
        "final_core_loss": metrics["final_core_loss"],
        "final_student_distill_loss": metrics["final_student_distill_loss"],
        "best_teacher_surface_loss": best_teacher_loss,
        "best_student_distill_loss": best_student_loss,
        "eval_teacher_surface_mse": evaluation_metrics["teacher_surface_mse"],
        "eval_teacher_pde_mse": evaluation_metrics["teacher_pde_mse"],
        "eval_teacher_ic_mse": evaluation_metrics["teacher_ic_mse"],
        "eval_teacher_bc_mse": evaluation_metrics["teacher_bc_mse"],
        "eval_teacher_center_mse": evaluation_metrics["teacher_center_mse"],
        "eval_teacher_core_mse": evaluation_metrics["teacher_core_mse"],
        "eval_student_mse": evaluation_metrics["student_mse"],
        "eval_student_temp_mse": evaluation_metrics["student_temp_mse"],
        "eval_student_cond_mse": evaluation_metrics["student_cond_mse"],
        "teacher_params": metrics["teacher_params"],
        "student_params": metrics["student_params"],
        "training_time_seconds": metrics["training_time_seconds"]
    }
    pd.DataFrame([flat_metrics]).to_csv(os.path.join(exp_dir, "metrics.csv"), index=False)

    # Append to global experiments_history.csv in output_dir
    global_csv = os.path.join(output_dir, "experiments_history.csv")
    global_df = pd.DataFrame([flat_metrics])
    if os.path.exists(global_csv):
        try:
            prev_df = pd.read_csv(global_csv)
            prev_df = prev_df[prev_df["exp_id"] != config.exp_id]
            global_df = pd.concat([prev_df, global_df], ignore_index=True)
        except Exception:
            pass
    global_df.to_csv(global_csv, index=False)

    # Generate and save convergence curves
    if save_plots:
        plot_path = os.path.join(exp_dir, "convergence_curves.png")
        plot_experiment_convergence(
            teacher_history=teacher_history,
            student_history=student_history,
            config=config,
            metrics=metrics,
            save_path=plot_path
        )

    return metrics
