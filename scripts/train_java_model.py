"""
SmellPredict — Train Java Defect Prediction Model CLI Entry Point
==================================================================
"""

import argparse
from pathlib import Path
from smellpredict.models.java_trainer import run_java_training_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmellPredict Java Model Training Pipeline")
    parser.add_argument("--data", type=Path, default=None, help="Path to parquet or duckdb")
    parser.add_argument("--output", type=Path, default=Path("models"), help="Output directory")
    parser.add_argument("--trials", type=int, default=50, help="Optuna HPO trials")
    args = parser.parse_args()

    run_java_training_pipeline(data_path=args.data, output_dir=args.output, n_trials=args.trials)
