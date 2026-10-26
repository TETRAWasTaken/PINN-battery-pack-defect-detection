import argparse
import sys
import pathlib

# Ensure root directory is on Python path
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import torch

from config.model_config import ModelConfig, get_default_search_grid
from data.sampler import BatteryDataSampler
from training.experiment import run_single_experiment
from training.pipeline import run_batch_training_pipeline, select_device
from utils.paths import Paths

def parse_args():
    parser = argparse.ArgumentParser(
        description="PINN Battery Pack Defect Detection & Student Distillation Pipeline"
    )
    parser.add_argument(
        "--mode", choices=["batch", "single", "preprocess"], default="batch",
        help="Run hyperparameter search ('batch'), one experiment ('single'), or raw data preprocessing ('preprocess')"
    )
    parser.add_argument("--search", choices=["grid", "random"], default="random",
                        help="Batch search strategy")
    parser.add_argument("--n-trials", type=int, default=10,
                        help="Number of random configurations in random search")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs-teacher", type=int, default=20, help="Number of teacher training epochs")
    parser.add_argument("--epochs-student", type=int, default=10, help="Number of student distillation epochs")
    parser.add_argument("--n-surface", type=int, default=150, help="Number of surface sensor points")
    parser.add_argument("--n-collocation", type=int, default=800, help="Number of interior collocation points")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cuda', 'mps', 'cpu')")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for saved models and metrics")

    # Single-run specific arguments
    parser.add_argument("--exp-id", type=str, default="single_run", help="Experiment ID for single run")
    parser.add_argument("--lr-teacher", type=float, default=1e-3, help="Teacher learning rate")
    parser.add_argument("--lr-student", type=float, default=1e-3, help="Student learning rate")
    parser.add_argument("--teacher-hidden-dim", type=int, default=128, help="Teacher hidden layer size")
    parser.add_argument("--fourier-scale", type=float, default=5.0, help="Fourier feature projection scale")
    parser.add_argument("--lambda-pde", type=float, default=0.01, help="Physics loss weighting factor")
    parser.add_argument("--lambda-surf", type=float, default=1.0, help="Surface sensor loss weighting factor")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device(args.device)
    output_dir = args.output_dir or str(Paths.experiments_dir())

    if args.mode == "preprocess":
        from data.preprocess import process_all_nasa_batteries
        print("Starting Raw Data Preprocessing Pipeline...")
        process_all_nasa_batteries(output_dir=output_dir)
    elif args.mode == "batch":
        print("Starting Batch Training Pipeline...")
        run_batch_training_pipeline(
            grid=get_default_search_grid(),
            output_dir=output_dir,
            device=device,
            epochs_teacher=args.epochs_teacher,
            epochs_student=args.epochs_student,
            n_surface=args.n_surface,
            n_collocation=args.n_collocation,
            search=args.search,
            n_trials=args.n_trials,
            seed=args.seed,
        )
    else:
        print(f"Starting Single Experiment: {args.exp_id}...")
        config = ModelConfig(
            exp_id=args.exp_id,
            lr_teacher=args.lr_teacher,
            lr_student=args.lr_student,
            teacher_hidden_dim=args.teacher_hidden_dim,
            fourier_scale=args.fourier_scale,
            lambda_pde=args.lambda_pde,
            lambda_surf=args.lambda_surf,
            epochs_teacher=args.epochs_teacher,
            epochs_student=args.epochs_student,
            batch_size=256,
            seed=args.seed,
        )
        sampler = BatteryDataSampler(n_surface=args.n_surface, n_collocation=args.n_collocation)
        metrics = run_single_experiment(config, sampler, device, output_dir)
        print(f"\nExperiment {args.exp_id} completed successfully!")
        print(f"Final Teacher Loss: {metrics['final_teacher_loss']:.4f}")
        print(f"Final Student Distillation Loss: {metrics['final_student_distill_loss']:.6f}")


if __name__ == "__main__":
    main()
