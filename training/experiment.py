import os
import json
import time
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

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


def run_single_experiment(config: ModelConfig,
                          sampler: BatteryDataSampler,
                          device: torch.device,
                          output_dir: str) -> Dict[str, Any]:
    """
    Executes a single experiment configuration:
      1. Trains the Teacher Fourier-SIREN PINN with Surface Data + Heat PDE Residuals.
      2. Distills knowledge from Teacher PINN into the lightweight Student MLP.
      3. Saves checkpoints (teacher_pinn.pt, student_mlp.pt) and metrics.json.
    """
    print(f"\n==================================================")
    print(f" Running Experiment: {config.exp_id}")
    print(f" Config: LR_T={config.lr_teacher}, Hidden={config.teacher_hidden_dim}, "
          f"FourierScale={config.fourier_scale}, LambdaPDE={config.lambda_pde}")
    print(f"==================================================")

    start_time = time.time()
    set_seed(config.seed)

    # 1. Instantiate & Train Teacher PINN
    teacher = TeacherPINN(
        hidden_dim=config.teacher_hidden_dim,
        scale=config.fourier_scale
    ).to(device)
    optimizer_teacher = optim.Adam(teacher.parameters(), lr=config.lr_teacher)

    teacher_loss_history = []
    best_teacher_loss = float("inf")
    best_teacher_state = None
    for epoch in range(1, config.epochs_teacher + 1):
        coords_surf, temp_surf, coords_coll = sampler.sample_data(device)

        optimizer_teacher.zero_grad()

        # Surface Sensor Loss
        pred_surf_temp, _ = teacher(coords_surf)
        loss_surf = nn.functional.mse_loss(pred_surf_temp, temp_surf)

        # Physics-informed PDE residual loss
        loss_pde = compute_pde_residual(teacher, coords_coll)

        # Total Weighted Objective
        total_loss = config.lambda_surf * loss_surf + config.lambda_pde * loss_pde
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), max_norm=1.0)
        optimizer_teacher.step()

        teacher_loss_history.append(total_loss.item())
        if total_loss.item() < best_teacher_loss:
            best_teacher_loss = total_loss.item()
            best_teacher_state = {k: v.detach().cpu().clone() for k, v in teacher.state_dict().items()}

        if epoch % max(1, config.epochs_teacher // 4) == 0 or epoch == 1:
            print(f" [Teacher Epoch {epoch:03d}/{config.epochs_teacher}] "
                  f"Total Loss: {total_loss.item():.4f} | "
                  f"Surf Loss: {loss_surf.item():.4f} | "
                  f"PDE Loss: {loss_pde.item():.4f}")

    # 2. Knowledge Distillation to Student MLP
    print("\n -> Starting Knowledge Distillation to Student MLP...")
    student = StudentMLP().to(device)
    optimizer_student = optim.Adam(student.parameters(), lr=config.lr_student)

    student_loss_history = []
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

        student_loss_history.append(distill_loss.item())
        if distill_loss.item() < best_student_loss:
            best_student_loss = distill_loss.item()
            best_student_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}

        if epoch % max(1, config.epochs_student // 2) == 0 or epoch == 1:
            print(f" [Student Epoch {epoch:03d}/{config.epochs_student}] "
                  f"Distillation Loss: {distill_loss.item():.6f}")

    if best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)
    if best_student_state is not None:
        student.load_state_dict(best_student_state)

    # Use a separate seeded batch so evaluation is not the final training batch.
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
        "final_teacher_loss": teacher_loss_history[-1],
        "final_student_distill_loss": student_loss_history[-1],
        "best_teacher_loss": best_teacher_loss,
        "best_student_distill_loss": best_student_loss,
        "evaluation": evaluation_metrics,
        "teacher_params": sum(p.numel() for p in teacher.parameters()),
        "student_params": sum(p.numel() for p in student.parameters()),
        "training_time_seconds": round(elapsed_time, 2)
    }

    with open(os.path.join(exp_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
