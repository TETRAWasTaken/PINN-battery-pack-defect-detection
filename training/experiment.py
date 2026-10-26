import os
import json
import time
from typing import Dict, Any, Tuple, List

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
from physics.heat_equation import compute_pde_residual


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_t = teacher_history["epoch"]
    total_loss = teacher_history["total_loss"]
    surf_loss = teacher_history["surf_loss"]
    pde_loss = teacher_history["pde_loss"]

    # --- Left Panel: Teacher PINN Convergence ---
    color_total = "#1f77b4"
    color_surf = "#2ca02c"
    color_pde = "#d62728"

    line1 = ax1.plot(epochs_t, total_loss, label="Total Weighted Loss", color=color_total, linewidth=2)
    line2 = ax1.plot(epochs_t, surf_loss, label="Surface Sensor Loss (MSE)", color=color_surf, linestyle="--", linewidth=1.8)
    ax1.set_xlabel("Teacher Epoch", fontsize=11, fontweight="medium")
    ax1.set_ylabel("Total & Surface Loss", fontsize=11, color=color_total)
    ax1.tick_params(axis="y", labelcolor=color_total)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)

    # Use twinx for PDE residual to handle scale differences cleanly
    ax1_twin = ax1.twinx()
    line3 = ax1_twin.plot(epochs_t, pde_loss, label=f"PDE Residual Loss", color=color_pde, linestyle=":", linewidth=2)
    ax1_twin.set_ylabel("PDE Residual Loss (MSE)", fontsize=11, color=color_pde)
    ax1_twin.tick_params(axis="y", labelcolor=color_pde)
    ax1_twin.set_yscale("log")

    # Combined legend
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper right", framealpha=0.9)
    ax1.set_title(f"Teacher PINN Convergence (Final Surface MSE: {surf_loss[-1]:.4f})", fontsize=12, fontweight="bold")

    # --- Right Panel: Student MLP Distillation ---
    epochs_s = student_history["epoch"]
    distill_loss = student_history["distill_loss"]
    color_distill = "#9467bd"

    ax2.plot(epochs_s, distill_loss, label="Distillation Loss (MSE)", color=color_distill, linewidth=2, marker="o", markersize=4)
    ax2.set_xlabel("Student Epoch", fontsize=11, fontweight="medium")
    ax2.set_ylabel("Distillation Loss (MSE)", fontsize=11, color=color_distill)
    ax2.tick_params(axis="y", labelcolor=color_distill)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend(loc="upper right", framealpha=0.9)
    ax2.set_title(f"Student Distillation (Final MSE: {distill_loss[-1]:.6f})", fontsize=12, fontweight="bold")

    fig.suptitle(
        f"Experiment: {config.exp_id} | LR_T={config.lr_teacher:.2e} | Hidden={config.teacher_hidden_dim} | "
        f"Scale(S/T)={config.scale_space or config.fourier_scale:.1f}/{config.scale_time or config.fourier_scale:.1f} | "
        f"Time: {metrics['training_time_seconds']}s",
        fontsize=12,
        y=1.02,
        fontweight="bold"
    )

    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Convergence graphs saved to: {save_path}")


def run_single_experiment(config: ModelConfig,
                          sampler: BatteryDataSampler,
                          device: torch.device,
                          output_dir: str,
                          save_plots: bool = True) -> Dict[str, Any]:
    """
    Executes a single experiment configuration with improved PINN dynamics:
      1. Dimensionless coordinate normalization.
      2. Ambient temperature residual formulation.
      3. Anisotropic Fourier feature frequencies.
      4. Cosine Annealing learning rate schedule.
      5. Optional adaptive loss balancing.
    """
    print(f"\n==================================================")
    print(f" Running Experiment: {config.exp_id}")
    print(f" Config: LR_T={config.lr_teacher:.2e}, Hidden={config.teacher_hidden_dim}, "
          f"FourierScale={config.fourier_scale}, LambdaPDE={config.lambda_pde}")
    print(f"==================================================")

    start_time = time.time()
    set_seed(config.seed)

    # 1. Instantiate Teacher PINN with anisotropic multi-scale support
    teacher = TeacherPINN(
        hidden_dim=config.teacher_hidden_dim,
        scale=config.fourier_scale,
        scale_space=config.scale_space,
        scale_time=config.scale_time,
        coordinate_scale=(0.009, 0.009, 0.065, 3600.0),
        t_ambient=297.15
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
        "lr": []
    }
    best_teacher_loss = float("inf")
    best_teacher_state = None

    for epoch in range(1, config.epochs_teacher + 1):
        coords_surf, temp_surf, coords_coll = sampler.sample_data(device)

        # Surface Sensor Loss
        pred_surf_temp, _ = teacher(coords_surf)
        loss_surf = nn.functional.mse_loss(pred_surf_temp, temp_surf)

        # Physics-informed PDE residual loss
        loss_pde = compute_pde_residual(teacher, coords_coll)

        if config.use_adaptive_loss:
            # Relative gradient magnitude balancing
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
        total_loss = config.lambda_surf * loss_surf + effective_lambda_pde * loss_pde
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
        teacher_history["lr"].append(current_lr)

        if loss_surf.item() < best_teacher_loss:
            best_teacher_loss = loss_surf.item()
            best_teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}

        if epoch % max(1, config.epochs_teacher // 5) == 0 or epoch == 1:
            print(f" [Teacher Epoch {epoch:03d}/{config.epochs_teacher}] "
                  f"Total Loss: {total_loss.item():.4f} | "
                  f"Surf Loss: {loss_surf.item():.4f} | "
                  f"PDE Loss: {loss_pde.item():.4f} | LR: {current_lr:.2e}")

    # 2. Knowledge Distillation to Student MLP
    print("\n -> Starting Knowledge Distillation to Student MLP...")
    student = StudentMLP(
        coordinate_scale=(0.009, 0.009, 0.065, 3600.0),
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

    for epoch in range(1, config.epochs_student + 1):
        coords_surf, _, coords_coll = sampler.sample_data(device)
        eval_coords = torch.cat([coords_surf, coords_coll], dim=0)

        with torch.no_grad():
            t_temp, t_cond = teacher(eval_coords)
            teacher_targets = torch.cat([t_temp, t_cond], dim=-1)

        optimizer_student.zero_grad()
        student_preds = student(eval_coords)
        distill_loss = nn.functional.mse_loss(student_preds, teacher_targets)
        distill_loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
        optimizer_student.step()

        if scheduler_student is not None:
            scheduler_student.step()

        student_history["epoch"].append(epoch)
        student_history["distill_loss"].append(distill_loss.item())

        if distill_loss.item() < best_student_loss:
            best_student_loss = distill_loss.item()
            best_student_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        if epoch % max(1, config.epochs_student // 3) == 0 or epoch == 1:
            print(f" [Student Epoch {epoch:03d}/{config.epochs_student}] "
                  f"Distillation Loss: {distill_loss.item():.6f}")

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
        "epochs_teacher": config.epochs_teacher,
        "epochs_student": config.epochs_student,
        "final_teacher_loss": metrics["final_teacher_loss"],
        "final_surface_loss": metrics["final_surface_loss"],
        "final_pde_loss": metrics["final_pde_loss"],
        "final_student_distill_loss": metrics["final_student_distill_loss"],
        "best_teacher_surface_loss": best_teacher_loss,
        "best_student_distill_loss": best_student_loss,
        "eval_teacher_surface_mse": evaluation_metrics["teacher_surface_mse"],
        "eval_teacher_pde_mse": evaluation_metrics["teacher_pde_mse"],
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
