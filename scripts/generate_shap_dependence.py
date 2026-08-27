"""
SmellPredict — Generate SHAP Dependence Plots for Top Features
==============================================================
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path
import shap

from smellpredict.models.trainer import ExperimentRunner

def generate_dependence():
    figures_dir = Path("reports/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    best_model_path = Path("models/best_model.pkl")
    merged_path = Path("data/raw/all_repos_merged.parquet")
    
    with open(best_model_path, "rb") as f:
        calibrated_model = pickle.load(f)
    
    df_all = pd.read_parquet(merged_path)
    runner = ExperimentRunner(df=df_all)
    feature_cols = runner._get_feature_cols("B")
    
    X_sample = df_all[feature_cols].fillna(0).sample(n=min(1000, len(df_all)), random_state=42)
    
    underlying_pipeline = calibrated_model.calibrated_classifiers_[0].estimator
    preprocessor = underlying_pipeline.named_steps.get("preprocessor")
    core_model = underlying_pipeline.named_steps.get("model")
    
    X_proc = preprocessor.transform(X_sample.values) if preprocessor else X_sample.values
    explainer = shap.TreeExplainer(core_model)
    shap_vals = explainer.shap_values(X_proc)
    if isinstance(shap_vals, list) and len(shap_vals) == 2:
        shap_vals = shap_vals[1]
    
    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
        # shape: (n_samples, n_features, n_classes)
        shap_vals = shap_vals[:, :, 1]

    mean_shap = np.mean(np.abs(shap_vals), axis=0)
    top_indices = [int(x) for x in np.argsort(mean_shap)[::-1][:5]]
    top_features = [feature_cols[i] for i in top_indices]
    
    print(f"Top 5 SHAP features: {top_features}")
    
    for rank, (feat_idx, f_name) in enumerate(zip(top_indices, top_features), 1):
        plt.figure(figsize=(8, 6))
        shap.dependence_plot(
            feat_idx,
            shap_vals,
            features=X_sample.values,
            feature_names=feature_cols,
            show=False,
        )
        plt.title(f"Figure 1.{rank}: SHAP Dependence Plot — {f_name}", fontsize=13, pad=10)
        plt.tight_layout()
        save_file = figures_dir / f"shap_dependence_{f_name}.png"
        plt.savefig(save_file, dpi=200)
        plt.close()
        print(f"  Saved dependence plot: {save_file}")

if __name__ == "__main__":
    generate_dependence()
