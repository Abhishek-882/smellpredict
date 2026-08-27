"""
run_eval_4repo_ood.py — 4-Repo Zero-Shot OOD Benchmark on Trained Dual Engines
==============================================================================
Evaluates Engine A and Engine B on 4 diverse holdout repositories:
  - django (78.11% prev, Web ORM / Framework)
  - rich (74.29% prev, Terminal UI / Formatting)
  - pillow (41.82% prev, Imaging / C-Extension)
  - fastapi (20.91% prev, Modern Async Web API)
Exports to data/processed/experiment_results_dual_engine_v12.csv.
"""

import sys
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    auc,
    brier_score_loss,
    average_precision_score
)

sys.path.insert(0, "src")
from smellpredict.models.trainer import LABEL_COL

src = "data/processed/enriched_v5.parquet"
df = pd.read_parquet(src)

meta_b = json.load(open("models/best_model_final_metadata.json"))
meta_a = json.load(open("models/engine_a_ast_static_metadata.json"))

features_b = meta_b["features"]
features_a = meta_a["features"]

model_b = joblib.load("models/best_model_final.pkl")
model_a = joblib.load("models/engine_a_ast_static.pkl")

RESERVED_OOD_REPOS = ["django", "rich", "pillow", "fastapi"]
test_df = df[df["repo"].isin(RESERVED_OOD_REPOS)].copy().reset_index(drop=True)
test_df = test_df[test_df["future_bug_fix"].isin([0, 1])].copy().reset_index(drop=True)
test_df["future_bug_fix"] = test_df["future_bug_fix"].astype(int)

def evaluate_repo(y_true, y_prob):
    n_pos = int(y_true.sum())
    n_neg = int(len(y_true) - n_pos)
    prevalence = float(y_true.mean())

    if n_pos == 0 or n_neg == 0:
        return {
            "n_rows": len(y_true), "n_pos": n_pos, "n_neg": n_neg,
            "prevalence": round(prevalence * 100, 2),
            "pr_auc": round(prevalence, 4), "roc_auc": 0.5,
            "brier": round(float(brier_score_loss(y_true, y_prob)), 4),
            "lift": 0.0, "prec_at_20": round(prevalence * 100, 2), "rec_at_20": 20.0,
        }

    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    lift = pr_auc - prevalence

    sorted_indices = np.argsort(-y_prob)
    k20 = max(1, int(len(y_true) * 0.20))
    top20_idx = sorted_indices[:k20]
    prec_at_20 = float(y_true[top20_idx].mean())
    rec_at_20 = float(y_true[top20_idx].sum() / n_pos) if n_pos > 0 else 0.0

    return {
        "n_rows": len(y_true), "n_pos": n_pos, "n_neg": n_neg,
        "prevalence": round(prevalence * 100, 2),
        "pr_auc": round(pr_auc, 4), "roc_auc": round(roc_auc, 4),
        "brier": round(brier, 4), "lift": round(lift, 4),
        "prec_at_20": round(prec_at_20 * 100, 2), "rec_at_20": round(rec_at_20 * 100, 2),
    }

ood_records = []
y_ood = test_df[LABEL_COL].values
prob_ood_b = model_b.predict_proba(test_df[features_b].fillna(0))[:, 1]
prob_ood_a = model_a.predict_proba(test_df[features_a].fillna(0))[:, 1]

domain_map = {
    "django": "Web ORM",
    "rich": "Terminal UI",
    "pillow": "Imaging C-Ext",
    "fastapi": "Async Web API"
}

for rname in RESERVED_OOD_REPOS:
    sub = test_df[test_df["repo"] == rname]
    y_sub = sub[LABEL_COL].values
    p_b = model_b.predict_proba(sub[features_b].fillna(0))[:, 1]
    p_a = model_a.predict_proba(sub[features_a].fillna(0))[:, 1]

    eval_b = evaluate_repo(y_sub, p_b)
    eval_a = evaluate_repo(y_sub, p_a)

    ood_records.append({
        "repo": rname,
        "domain": domain_map.get(rname, "General"),
        "n_rows": eval_b["n_rows"],
        "n_pos": eval_b["n_pos"],
        "n_neg": eval_b["n_neg"],
        "prevalence_pct": eval_b["prevalence"],
        "engine_b_pr_auc": eval_b["pr_auc"],
        "engine_b_roc_auc": eval_b["roc_auc"],
        "engine_b_brier": eval_b["brier"],
        "engine_b_lift": eval_b["lift"],
        "engine_b_prec20": eval_b["prec_at_20"],
        "engine_b_rec20": eval_b["rec_at_20"],
        "engine_a_pr_auc": eval_a["pr_auc"],
        "engine_a_roc_auc": eval_a["roc_auc"],
        "engine_a_brier": eval_a["brier"],
        "engine_a_lift": eval_a["lift"],
        "engine_a_prec20": eval_a["prec_at_20"],
        "engine_a_rec20": eval_a["rec_at_20"],
    })

# Combined 4-Repo OOD
comb_b = evaluate_repo(y_ood, prob_ood_b)
comb_a = evaluate_repo(y_ood, prob_ood_a)

ood_records.append({
    "repo": "COMBINED_4_REPO_OOD",
    "domain": "Pooled Zero-Shot Pool",
    "n_rows": comb_b["n_rows"],
    "n_pos": comb_b["n_pos"],
    "n_neg": comb_b["n_neg"],
    "prevalence_pct": comb_b["prevalence"],
    "engine_b_pr_auc": comb_b["pr_auc"],
    "engine_b_roc_auc": comb_b["roc_auc"],
    "engine_b_brier": comb_b["brier"],
    "engine_b_lift": comb_b["lift"],
    "engine_b_prec20": comb_b["prec_at_20"],
    "engine_b_rec20": comb_b["rec_at_20"],
    "engine_a_pr_auc": comb_a["pr_auc"],
    "engine_a_roc_auc": comb_a["roc_auc"],
    "engine_a_brier": comb_a["brier"],
    "engine_a_lift": comb_a["lift"],
    "engine_a_prec20": comb_a["prec_at_20"],
    "engine_a_rec20": comb_a["rec_at_20"],
})

ood_df = pd.DataFrame(ood_records)
ood_out = "data/processed/experiment_results_dual_engine_v12.csv"
ood_df.to_csv(ood_out, index=False)
print("Saved 4-Repo OOD benchmark results ->", ood_out)
print(ood_df.to_string())
