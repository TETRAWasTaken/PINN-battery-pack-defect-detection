import os
import json
import itertools
from typing import Dict, Any, List, Optional

import torch
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config.model_config import ModelConfig, get_default_search_grid, sample_random_configs
from data.sampler import BatteryDataSampler
from training.experiment import run_single_experiment
from utils.paths import Paths


def select_device(preferred: Optional[str] = None) -> torch.device:
    """Selects the best available accelerator device (CUDA, Apple MPS, or CPU)."""
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def plot_batch_comparison(all_results: List[Dict[str, Any]], output_dir: str):
    """Generates an overarching comparative visualization across all sweep configurations."""
    if not all_results:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Load history CSVs for each experiment
    cmap = plt.get_cmap("tab10")
    for i, res in enumerate(all_results):
        exp_id = res["exp_id"]
        color = cmap(i % 10)
        hist_path = os.path.join(output_dir, exp_id, "training_history.csv")

        if os.path.exists(hist_path):
            try:
                hist_df = pd.read_csv(hist_path)
                # Filter valid teacher and student rows
                t_df = hist_df.dropna(subset=["teacher_total_loss"])
                s_df = hist_df.dropna(subset=["student_distill_loss"])

                label = f"{exp_id} (LR={res['config']['lr_teacher']}, H={res['config']['teacher_hidden_dim']})"
                ax1.plot(t_df["epoch"], t_df["teacher_total_loss"], label=label, color=color, linewidth=1.8, alpha=0.85)
                ax2.plot(s_df["epoch"], s_df["student_distill_loss"], label=exp_id, color=color, linewidth=1.8, marker="o", markersize=3, alpha=0.85)
            except Exception:
                pass

    ax1.set_title("Teacher Total Loss Convergence Across Models", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Total Loss (log scale)", fontsize=11)
    ax1.set_yscale("log")
    ax1.grid(True, which="both", linestyle="--", alpha=0.3)
    ax1.legend(fontsize=8, loc="upper right", framealpha=0.85)

    ax2.set_title("Student Distillation Loss Convergence Across Models", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Distillation Loss (log scale)", fontsize=11)
    ax2.set_yscale("log")
    ax2.grid(True, which="both", linestyle="--", alpha=0.3)
    ax2.legend(fontsize=8, loc="upper right", framealpha=0.85)

    plt.tight_layout()
    comp_path = os.path.join(output_dir, "batch_convergence_comparison.png")
    fig.savefig(comp_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f" -> Batch convergence comparison plot saved to: {comp_path}")


def run_batch_training_pipeline(
    grid: Optional[Dict[str, List[Any]]] = None,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    epochs_teacher: int = 20,
    epochs_student: int = 10,
    n_surface: int = 150,
    n_collocation: int = 800,
    search: str = "grid",
    n_trials: int = 10,
    seed: int = 42,
    save_plots: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generates a hyperparameter grid and executes the sequential training pipeline.
    Exports JSON & CSV summary reports and convergence comparison graphs.
    """
    if device is None:
        device = select_device()
    print(f"Using Compute Device: {device}")

    if output_dir is None:
        output_dir = str(Paths.experiments_dir())
    os.makedirs(output_dir, exist_ok=True)

    if grid is None:
        grid = get_default_search_grid()

    if search == "random":
        combinations = sample_random_configs(n_trials=n_trials, seed=seed)
    else:
        keys, values = zip(*grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    print(f"Total Model Configurations to Train: {len(combinations)}")

    sampler = BatteryDataSampler(n_surface=n_surface, n_collocation=n_collocation)
    all_results: List[Dict[str, Any]] = []

    for idx, param_set in enumerate(combinations):
        config = ModelConfig(
            exp_id=f"exp_{idx+1:02d}",
            lr_teacher=param_set.get("lr_teacher", 1e-3),
            lr_student=param_set.get("lr_student", 1e-3),
            teacher_hidden_dim=param_set.get("teacher_hidden_dim", 128),
            fourier_scale=param_set.get("fourier_scale", 5.0),
            scale_space=param_set.get("scale_space"),
            scale_time=param_set.get("scale_time"),
            lambda_pde=param_set.get("lambda_pde", 0.01),
            lambda_surf=param_set.get("lambda_surf", 1.0),
            epochs_teacher=epochs_teacher,
            epochs_student=epochs_student,
            batch_size=256,
            seed=seed + idx,
            use_lr_scheduler=param_set.get("use_lr_scheduler", True),
            use_adaptive_loss=param_set.get("use_adaptive_loss", False),
        )

        metrics = run_single_experiment(config, sampler, device, output_dir, save_plots=save_plots)
        all_results.append(metrics)

    print("\n" + "=" * 76)
    print(" SUMMARY OF ALL TRAINED MODELS")
    print("=" * 76)
    print(f"{'EXP ID':<8} | {'LR Teacher':<10} | {'Hidden':<6} | {'Fourier Scale':<13} | {'PDE Loss':<10} | {'Student Loss':<12}")
    print("-" * 76)

    for res in all_results:
        cfg = res["config"]
        print(f"{res['exp_id']:<8} | {cfg['lr_teacher']:<10} | {cfg['teacher_hidden_dim']:<6} | "
              f"{cfg['fourier_scale']:<13} | {res['final_teacher_loss']:<10.4f} | {res['final_student_distill_loss']:<12.6f}")

    # Sort results by student evaluation MSE (best model first)
    all_results.sort(key=lambda result: result["evaluation"]["student_mse"])

    # 1. Save JSON reports
    summary_json_file = os.path.join(output_dir, "summary_report.json")
    with open(summary_json_file, "w") as f:
        json.dump(all_results, f, indent=4)

    best_json_file = os.path.join(output_dir, "best_experiment.json")
    with open(best_json_file, "w") as f:
        json.dump(all_results[0], f, indent=4)

    # 2. Save CSV summary report
    summary_rows = []
    for res in all_results:
        cfg = res["config"]
        ev = res["evaluation"]
        summary_rows.append({
            "exp_id": res["exp_id"],
            "lr_teacher": cfg["lr_teacher"],
            "lr_student": cfg["lr_student"],
            "teacher_hidden_dim": cfg["teacher_hidden_dim"],
            "fourier_scale": cfg["fourier_scale"],
            "lambda_pde": cfg["lambda_pde"],
            "lambda_surf": cfg["lambda_surf"],
            "epochs_teacher": cfg["epochs_teacher"],
            "epochs_student": cfg["epochs_student"],
            "seed": cfg.get("seed"),
            "final_teacher_loss": res["final_teacher_loss"],
            "final_surface_loss": res.get("final_surface_loss"),
            "final_pde_loss": res.get("final_pde_loss"),
            "final_student_distill_loss": res["final_student_distill_loss"],
            "best_teacher_loss": res.get("best_teacher_loss"),
            "best_student_distill_loss": res.get("best_student_distill_loss"),
            "eval_teacher_surface_mse": ev["teacher_surface_mse"],
            "eval_teacher_pde_mse": ev["teacher_pde_mse"],
            "eval_student_mse": ev["student_mse"],
            "eval_student_temp_mse": ev["student_temp_mse"],
            "eval_student_cond_mse": ev["student_cond_mse"],
            "teacher_params": res["teacher_params"],
            "student_params": res["student_params"],
            "training_time_seconds": res["training_time_seconds"],
        })

    summary_csv_file = os.path.join(output_dir, "summary_report.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv_file, index=False)

    best_csv_file = os.path.join(output_dir, "best_experiment.csv")
    pd.DataFrame([summary_rows[0]]).to_csv(best_csv_file, index=False)

    print(f"\n[✓] Reports successfully exported:")
    print(f"    - JSON summary: {summary_json_file}")
    print(f"    - CSV summary:  {summary_csv_file}")
    print(f"    - Best model:   {best_csv_file} (Exp: {all_results[0]['exp_id']})")

    # 3. Save comparative convergence plot
    if save_plots:
        plot_batch_comparison(all_results, output_dir)

    print(f"\nAll models trained successfully! Results saved to: {output_dir}")
    return all_results
