"""
SmellPredict — Empirical Threshold Sensitivity Evaluation
=========================================================
Computes empirical defect rates and predictive correlation across multiple
smell detection thresholds from genuine mined snapshot data.
Saves results to data/processed/sensitivity_results.json.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def run_sensitivity():
    parquet_path = Path("data/raw/all_repos_merged.parquet")
    if not parquet_path.exists():
        print("Dataset not found!")
        return

    print("Loading empirical dataset for sensitivity analysis...")
    df = pd.read_parquet(parquet_path)
    
    thresholds_map = {
        "Long Method": {
            "col": "code_loc",
            "thresholds": [30, 40, 50, 60, 80],
        },
        "Long Param List": {
            "col": "code_sloc",  # proxy / correlate
            "thresholds": [10, 20, 30, 50, 100],
        },
        "Large Class": {
            "col": "code_loc",
            "thresholds": [100, 200, 300, 500, 1000],
        },
        "Deep Nesting": {
            "col": "code_cyclomatic_complexity",
            "thresholds": [3, 5, 8, 12, 20],
        },
        "High Complexity": {
            "col": "code_cyclomatic_complexity",
            "thresholds": [7, 10, 15, 20, 30],
        },
    }

    results = {}
    y = df["future_bug_fix"].values

    for smell_name, config in thresholds_map.items():
        col = config["col"]
        thresholds = config["thresholds"]
        smell_res = []

        if col in df.columns:
            feat_vals = df[col].fillna(0).values
            for t in thresholds:
                smell_active = (feat_vals >= t).astype(int)
                # Compute empirical bug rate among smelling files vs non-smelling
                n_active = int(np.sum(smell_active))
                if n_active > 0:
                    bug_rate_active = float(np.mean(y[smell_active == 1]))
                else:
                    bug_rate_active = 0.0

                # Compute point biserial correlation with bug fix
                corr = float(np.corrcoef(smell_active, y)[0, 1]) if np.std(smell_active) > 0 else 0.0
                
                smell_res.append({
                    "threshold": t,
                    "active_snapshots": n_active,
                    "active_bug_rate": round(bug_rate_active * 100, 2),
                    "defect_correlation": round(corr, 4),
                })
        results[smell_name] = smell_res

    out_file = Path("data/processed/sensitivity_results.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Empirical sensitivity analysis saved to {out_file}")


if __name__ == "__main__":
    run_sensitivity()
