import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, Any, List
import numpy as np
import torch

@dataclass
class ModelConfig:
    """Configuration dataclass for individual PINN and Student distillation experiments."""
    exp_id: str
    lr_teacher: float = 1e-3
    lr_student: float = 1e-3
    teacher_hidden_dim: int = 128
    fourier_scale: float = 5.0
    lambda_pde: float = 0.01
    lambda_surf: float = 1.0
    epochs_teacher: int = 20
    epochs_student: int = 10
    batch_size: int = 256
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def set_seed(seed: int = 42):
    """Sets random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends, "mps"):
        torch.mps.manual_seed(seed)


def get_default_search_grid() -> Dict[str, List[Any]]:
    """Returns the standard hyperparameter sweep grid for battery PINN training."""
    return {
        "lr_teacher": [1e-3, 5e-4],
        "teacher_hidden_dim": [64, 128],
        "fourier_scale": [3.0, 10.0],
        "lambda_pde": [0.01, 0.1]
    }


def sample_random_configs(n_trials: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample reproducible hyperparameter configurations without exhaustive enumeration."""
    if n_trials < 1:
        raise ValueError("n_trials must be at least 1")

    rng = random.Random(seed)
    configs = []
    for _ in range(n_trials):
        configs.append({
            "lr_teacher": 10 ** rng.uniform(-4.0, -2.5),
            "lr_student": 10 ** rng.uniform(-4.0, -2.5),
            "teacher_hidden_dim": rng.choice([64, 96, 128, 192]),
            "fourier_scale": rng.uniform(2.0, 12.0),
            "lambda_pde": 10 ** rng.uniform(-3.0, -0.5),
            "lambda_surf": 10 ** rng.uniform(-0.5, 0.5),
        })
    return configs
