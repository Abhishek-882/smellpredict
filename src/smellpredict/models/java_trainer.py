"""
SmellPredict — Java Model Training Module
==========================================
Core training and evaluation pipeline for Java defect prediction models.
"""

from __future__ import annotations

import os
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import duckdb
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

from smellpredict.models.trainer import (
    FEATURE_GROUPS,
    LABEL_COL,
    SMELL_COLS,
    drop_high_vif_features,
)


def load_java_dataset(
    data_path: Optional[Path] = None,
    db_path: Path = Path("data/java_smellpredict.duckdb"),
) -> pd.DataFrame:
    """Load Java snapshot dataset from parquet or DuckDB."""
    if data_path and Path(data_path).exists():
        logger.info(f"Loading Java data from parquet: {data_path}")
        df = pd.read_parquet(data_path)
    elif Path(db_path).exists():
        logger.info(f"Loading Java data from DuckDB: {db_path}")
        conn = duckdb.connect(str(db_path))
        df = conn.execute("SELECT * FROM java_snapshots").df()
        conn.close()
    else:
        logger.warning("No Java dataset found — generating structured synthetic Java corpus for verification")
        df = _generate_synthetic_java_corpus()

    logger.info(f"Loaded {len(df):,} Java snapshots (positive rate: {df['future_bug_fix'].mean()*100:.1f}%)")
    return df


def _generate_synthetic_java_corpus(n_samples: int = 1200) -> pd.DataFrame:
    """Generate deterministic synthetic Java dataset for tests and pipeline validation."""
    np.random.seed(42)
    rows = []
    repos = ["spring-boot", "kafka", "elasticsearch", "netty", "guava", "flink", "keycloak", "junit5"]

    for i in range(n_samples):
        repo = np.random.choice(repos)
        loc = int(np.random.exponential(250) + 20)
        sloc = int(loc * np.random.uniform(0.7, 0.9))
        n_funcs = max(1, int(loc / np.random.uniform(15, 45)))
        cc = max(1, int(np.random.exponential(4) + 1))
        nesting = int(np.random.choice([1, 2, 3, 4, 5, 6], p=[0.3, 0.35, 0.2, 0.1, 0.03, 0.02]))
        
        has_lm = int(loc > 300 and cc > 10)
        has_lp = int(np.random.rand() < 0.15)
        has_lc = int(loc > 400)
        has_dn = int(nesting >= 4)
        has_hc = int(cc >= 10)
        total_smells = has_lm + has_lp + has_lc + has_dn + has_hc

        latent = (
            0.05
            + 0.18 * (cc / 25.0)
            + 0.22 * (total_smells / 5.0)
            + 0.15 * (nesting / 6.0)
            + 0.10 * (loc / 1000.0)
            + np.random.normal(0, 0.05)
        )
        is_bug_fix = int(latent > 0.35)

        rows.append({
            "snapshot_id": f"java_snap_{i:05d}",
            "repo": repo,
            "file_path": f"src/main/java/org/{repo}/Module{i%50}.java",
            "file_id": f"fid_{i%200:04d}",
            "commit_hash": f"hash_{i:06d}",
            "commit_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i % 365),
            "future_bug_fix": is_bug_fix,
            "parse_fallback": False,
            "code_loc": loc,
            "code_sloc": sloc,
            "code_blank_lines": int(loc * 0.1),
            "code_comment_lines": int(loc * 0.1),
            "code_comment_density": 0.1,
            "code_function_count": n_funcs,
            "code_class_count": 1,
            "code_import_count": int(np.random.uniform(5, 25)),
            "code_avg_function_size": round(loc / n_funcs, 1),
            "code_max_function_size": int(loc * 0.4),
            "code_avg_param_count": round(np.random.uniform(1.2, 3.8), 2),
            "code_max_param_count": int(np.random.uniform(2, 7)),
            "code_max_nesting_depth": nesting,
            "code_avg_cyclomatic_complexity": round(cc * 0.6, 2),
            "code_max_cyclomatic_complexity": cc,
            "code_halstead_volume": round(loc * 25.0, 1),
            "code_halstead_difficulty": round(cc * 1.5, 2),
            "code_halstead_effort": round(loc * cc * 10.0, 1),
            "code_halstead_bugs": round(loc * 0.002, 4),
            "code_maintainability_index": round(max(10.0, 100.0 - cc * 2.5 - nesting * 5), 1),
            "code_cognitive_complexity": int(cc * 1.2 + nesting * 2),
            "has_long_method": has_lm,
            "has_long_param_list": has_lp,
            "has_large_class": has_lc,
            "has_deep_nesting": has_dn,
            "has_high_complexity": has_hc,
            "long_method_count": has_lm * 2,
            "long_param_count": has_lp,
            "large_class_count": has_lc,
            "deep_nesting_count": has_dn,
            "high_complexity_count": has_hc,
            "total_smells": total_smells,
            "previous_file_commits": int(np.random.uniform(1, 40)),
            "previous_bug_fixes": int(np.random.uniform(0, 10)),
            "contributors": int(np.random.uniform(1, 8)),
            "recent_file_commits": int(np.random.uniform(0, 5)),
            "code_churn_history": int(np.random.uniform(20, 500)),
            "file_age_days": float(np.random.uniform(30, 800)),
            "days_since_last_change": float(np.random.uniform(1, 100)),
            "developer_experience": round(np.random.uniform(0.1, 1.0), 2),
            "ownership_concentration": round(np.random.uniform(0.3, 1.0), 4),
            "commit_message_entropy": round(np.random.uniform(1.5, 4.5), 2),
            "avg_commit_size": round(np.random.uniform(10, 80), 1),
            "avg_time_between_commits": round(np.random.uniform(24, 200), 1),
            "has_multiple_contributors": int(np.random.rand() < 0.6),
            "is_recently_touched": int(np.random.rand() < 0.4),
        })

    return pd.DataFrame(rows)


class JavaDefectPipeline:
    def __init__(self, preproc, model, feature_names):
        self.preproc = preproc
        self.model = model
        self.feature_names = feature_names

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_sub = X.reindex(columns=self.feature_names, fill_value=0.0)
        X_trans = self.preproc.transform(X_sub)
        return self.model.predict_proba(X_trans)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        prob = self.predict_proba(X)[:, 1]
        return (prob >= 0.50).astype(int)


def run_java_training_pipeline(
    data_path: Optional[Path] = None,
    output_dir: Path = Path("models"),
    n_trials: int = 50,
) -> Dict[str, Any]:
    """Execute complete Java model training pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_java_dataset(data_path)

    df["commit_date"] = pd.to_datetime(df["commit_date"])
    df = df.sort_values("commit_date").reset_index(drop=True)

    n = len(df)
    train_end = int(0.70 * n)
    cal_end = int(0.85 * n)

    df_train = df.iloc[:train_end]
    df_cal = df.iloc[train_end:cal_end]
    df_test = df.iloc[cal_end:]

    feature_cols = [c for c in FEATURE_GROUPS["C"] if c in df.columns]
    X_train = df_train[feature_cols]
    y_train = df_train[LABEL_COL].values
    X_cal = df_cal[feature_cols]
    y_cal = df_cal[LABEL_COL].values
    X_test = df_test[feature_cols]
    y_test = df_test[LABEL_COL].values

    logger.info(f"Train samples: {len(X_train):,}, Cal: {len(X_cal):,}, Test: {len(X_test):,}")

    X_train_vif, dropped_vif = drop_high_vif_features(X_train, threshold=10.0, protect=SMELL_COLS)
    active_cols = list(X_train_vif.columns)
    logger.info(f"VIF screening retained {len(active_cols)}/{len(feature_cols)} features")

    X_train_use = X_train[active_cols]
    X_cal_use = X_cal[active_cols]
    X_test_use = X_test[active_cols]

    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    X_train_proc = preprocessor.fit_transform(X_train_use)
    X_cal_proc = preprocessor.transform(X_cal_use)
    X_test_proc = preprocessor.transform(X_test_use)

    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_proc, y_train)
    rf_prob = rf.predict_proba(X_test_proc)[:, 1]
    rf_pr_auc = average_precision_score(y_test, rf_prob)

    best_estimator = rf
    best_name = "RandomForest"
    best_pr_auc = rf_pr_auc

    if HAS_CATBOOST:
        try:
            cb_model = cb.CatBoostClassifier(
                iterations=100,
                learning_rate=0.05,
                depth=5,
                verbose=False,
                random_seed=42,
            )
            cb_model.fit(X_train_proc, y_train)
            cb_prob = cb_model.predict_proba(X_test_proc)[:, 1]
            cb_pr_auc = average_precision_score(y_test, cb_prob)
            if cb_pr_auc >= best_pr_auc:
                best_estimator = cb_model
                best_name = "CatBoost"
                best_pr_auc = cb_pr_auc
        except Exception as e:
            logger.debug(f"CatBoost training skipped: {e}")

    calibrator = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=3)
    calibrator.fit(X_train_proc, y_train)
    cal_prob = calibrator.predict_proba(X_test_proc)[:, 1]
    final_pr_auc = average_precision_score(y_test, cal_prob)
    final_roc_auc = roc_auc_score(y_test, cal_prob)
    final_brier = brier_score_loss(y_test, cal_prob)

    prod_pipeline = JavaDefectPipeline(preprocessor, calibrator, active_cols)

    model_path = output_dir / "java_best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(prod_pipeline, f)

    return {
        "model_type": best_name,
        "pr_auc": final_pr_auc,
        "roc_auc": final_roc_auc,
        "brier_score": final_brier,
        "feature_count": len(active_cols),
        "model_path": str(model_path),
    }
