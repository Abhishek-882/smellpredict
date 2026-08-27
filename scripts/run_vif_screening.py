"""
SmellPredict — Phase 3: VIF Multicollinearity Screening & Correlation Matrix
============================================================================
Runs Variance Inflation Factor screening on Feature Group C (all candidate features)
from the real 50-repository dataset, logs removed high-collinearity features,
and generates the correlation matrix heatmap.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from loguru import logger

from smellpredict.models.trainer import (
    FEATURE_GROUPS,
    CODE_METRIC_COLS,
    HISTORY_COLS,
    SMELL_COLS,
    compute_vif,
    drop_high_vif_features,
)

def run_vif_analysis(
    dataset_path: str = "data/raw/all_repos_merged.parquet",
    output_dir: str = "data/processed",
    figures_dir: str = "reports/figures",
    vif_threshold: float = 10.0,
):
    df = pd.read_parquet(dataset_path)
    print("=" * 70)
    print(f"PHASE 3: VIF SCREENING & FEATURE SELECTION (Dataset: {len(df):,} rows)")
    print("=" * 70)

    fg_c_cols = [c for c in FEATURE_GROUPS["C"] if c in df.columns]
    print(f"Total Feature Group C candidate features: {len(fg_c_cols)}")

    X_initial = df[fg_c_cols].fillna(0)

    # Initial VIF
    initial_vif = compute_vif(X_initial)
    print("\n--- Initial Top-15 VIF Values ---")
    print(initial_vif.head(15).to_string(index=False))

    # Protect code smell columns (essential to RQ1-RQ6)
    protected_smells = [c for c in SMELL_COLS if c in fg_c_cols]
    
    # Run iterative VIF dropping
    X_screened, removed_cols = drop_high_vif_features(
        X_initial, threshold=vif_threshold, protect=protected_smells
    )

    print(f"\n--- VIF Screening Summary (threshold = {vif_threshold}) ---")
    print(f"Features removed: {len(removed_cols)}")
    for col in removed_cols:
        v = initial_vif[initial_vif['feature'] == col]['vif'].values[0]
        print(f"  - {col:35s} (initial VIF: {v:.2f})")

    final_vif = compute_vif(X_screened)
    print(f"\nRemaining features after VIF: {len(X_screened.columns)}")
    print("\n--- Final VIF Values (Max VIF <= 10.0 for non-protected) ---")
    print(final_vif.head(15).to_string(index=False))

    # Save screened feature list and VIF reports
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    initial_vif.to_csv(out_path / "vif_initial.csv", index=False)
    final_vif.to_csv(out_path / "vif_screened.csv", index=False)
    
    with open(out_path / "screened_features.json", "w") as f:
        import json
        json.dump({
            "initial_features": fg_c_cols,
            "removed_features": removed_cols,
            "screened_features": list(X_screened.columns),
            "vif_threshold": vif_threshold,
        }, f, indent=2)

    # Generate and save correlation matrix heatmap
    fig_path = Path(figures_dir)
    fig_path.mkdir(parents=True, exist_ok=True)

    print("\nGenerating Correlation Matrix Heatmap...")
    corr = X_screened.corr()
    
    plt.figure(figsize=(16, 14), dpi=150)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmax=1.0, vmin=-1.0,
        center=0, square=True, linewidths=0.5,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"}
    )
    plt.title(f"Feature Correlation Matrix (After VIF Screening, N={len(X_screened.columns)})", fontsize=14, pad=15)
    plt.tight_layout()
    corr_plot_path = fig_path / "feature_correlation_heatmap.png"
    plt.savefig(corr_plot_path)
    plt.close()
    print(f"Saved correlation heatmap to: {corr_plot_path}")
    print("=" * 70)

if __name__ == "__main__":
    run_vif_analysis()
