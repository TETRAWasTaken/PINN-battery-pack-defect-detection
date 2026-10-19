import os
import json
import itertools
from typing import Dict, Any, List, Optional

import torch

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
) -> List[Dict[str, Any]]:
    """
    Generates a hyperparameter grid and executes the sequential training pipeline.
    
    Args:
        grid: Optional dictionary of hyperparameter lists. Defaults to standard 16-combination sweep.
        output_dir: Destination directory for artifacts. Defaults to Paths.experiments_dir().
        device: Torch compute device. Auto-selects CUDA > MPS > CPU if None.
        epochs_teacher: Teacher training epochs per experiment.
        epochs_student: Student distillation epochs per experiment.
        n_surface: Number of surface sensor samples per step.
        n_collocation: Number of interior volume collocation samples per step.
        
    Returns:
        List of result metric dictionaries for all experiments.
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
            lambda_pde=param_set.get("lambda_pde", 0.01),
            lambda_surf=param_set.get("lambda_surf", 1.0),
            epochs_teacher=epochs_teacher,
            epochs_student=epochs_student,
            batch_size=256,
            seed=seed + idx,
        )

        metrics = run_single_experiment(config, sampler, device, output_dir)
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

    summary_file = os.path.join(output_dir, "summary_report.json")
    all_results.sort(key=lambda result: result["evaluation"]["student_mse"])
    with open(summary_file, "w") as f:
        json.dump(all_results, f, indent=4)
    with open(os.path.join(output_dir, "best_experiment.json"), "w") as f:
        json.dump(all_results[0], f, indent=4)

    print(f"\nAll models trained successfully! Results saved to: {output_dir}")
    return all_results
