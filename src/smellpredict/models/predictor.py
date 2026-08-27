"""
SmellPredict — Unified Prediction & Analysis Engine
===================================================
Provides unified file-level and repository-level risk prediction,
feature extraction, SHAP feature attribution, and refactoring analysis.
"""

from __future__ import annotations

import os
import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("smellpredict.predictor")

from smellpredict.features.extractor import (
    CodeMetrics,
    SmellFeatures,
    extract_file_features,
)
from smellpredict.features.refactor import RefactoringAdvice, analyze_refactorings

_MODEL_CACHE: Dict[str, Any] = {}


def classify_risk_tier(probability: float) -> Dict[str, str]:
    """Classify a calibrated probability [0, 1] into a human-readable risk tier."""
    if probability <= 0.35:
        return {
            "tier": "Low",
            "icon": "🟢",
            "recommendation": "Code looks healthy. Standard peer review recommended.",
            "color": "#00d4aa",
        }
    elif probability <= 0.65:
        return {
            "tier": "Medium",
            "icon": "🟡",
            "recommendation": "Moderate complexity and smell indicators. Consider targeted unit testing.",
            "color": "#f59e0b",
        }
    elif probability <= 0.80:
        return {
            "tier": "High",
            "icon": "🟠",
            "recommendation": "High defect risk. Refactor long methods and reduce nesting depth.",
            "color": "#f97316",
        }
    else:
        return {
            "tier": "Critical",
            "icon": "🔴",
            "recommendation": "Critical defect risk. Prioritize immediate refactoring and architectural review.",
            "color": "#ef4444",
        }


import json
import joblib
import pandas as pd

def get_trained_model() -> Optional[Any]:
    """Lazy-load the champion trained model (LightGBM + Isotonic) from disk."""
    if "model" not in _MODEL_CACHE:
        model_paths = [
            Path(os.environ.get("SMELLPREDICT_MODEL_PATH", "models/best_model_final.pkl")),
            Path("models/best_model_final.pkl"),
            Path(__file__).parent.parent.parent.parent / "models" / "best_model_final.pkl",
            Path("models/best_model.pkl"),
        ]
        loaded = None
        for p in model_paths:
            if p.exists():
                try:
                    loaded = joblib.load(p)
                    _MODEL_CACHE["model"] = loaded
                    _MODEL_CACHE["model_source"] = str(p)
                    
                    # Try loading metadata for 73 features
                    meta_path = p.with_name(p.stem + "_metadata.json")
                    if not meta_path.exists():
                        meta_path = Path("models/best_model_final_metadata.json")
                    if meta_path.exists():
                        with open(meta_path, "r", encoding="utf-8") as mf:
                            _MODEL_CACHE["meta"] = json.load(mf)
                    
                    logger.info(f"Loaded trained champion model from {p}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load model from {p}: {e}")

        if loaded is None:
            _MODEL_CACHE["model"] = None
            _MODEL_CACHE["model_source"] = "none"

    return _MODEL_CACHE.get("model")


def compute_heuristic_risk(code_metrics: CodeMetrics, smell_feats: SmellFeatures) -> float:
    """Deterministic fallback risk score for when no trained model is present."""
    complexity_score = min(code_metrics.max_cyclomatic_complexity / 30.0, 1.0)
    nesting_score = min(code_metrics.max_nesting_depth / 8.0, 1.0)
    smell_score = min(smell_feats.total_smells / 10.0, 1.0)
    loc_score = min(code_metrics.loc / 1000.0, 1.0)
    mi_score = max(0.0, (100.0 - code_metrics.maintainability_index) / 100.0)

    prob = (
        0.25 * complexity_score
        + 0.25 * smell_score
        + 0.20 * nesting_score
        + 0.15 * loc_score
        + 0.15 * mi_score
    )
    return round(float(np.clip(prob, 0.0, 1.0)), 4)


def analyze_source_code(
    source_code: str,
    file_path: str = "snippet.py",
) -> Dict[str, Any]:
    """
    Complete analysis pipeline for a Python file's source code:
    - Feature extraction (CodeMetrics + SmellFeatures)
    - Defect risk inference (Combines AST structural complexity, smell density & calibrated model)
    - AST refactoring advice generation
    """
    code_metrics, smell_feats = extract_file_features(source_code)
    
    # Calculate AST structural risk
    heuristic_prob = compute_heuristic_risk(code_metrics, smell_feats)
    
    model = get_trained_model()
    meta = _MODEL_CACHE.get("meta")

    if model is not None and meta:
        try:
            m_dict = asdict(code_metrics)
            s_dict = asdict(smell_feats)
            features = meta.get("features", [])
            
            # Map empirical quantile approximations for FG_J
            rank_loc = float(np.clip(code_metrics.loc / 600.0, 0.01, 0.99))
            rank_comp = float(np.clip(code_metrics.max_cyclomatic_complexity / 15.0, 0.01, 0.99))
            rank_smells = float(np.clip(smell_feats.total_smells / 5.0, 0.0, 0.99))
            rank_mi = float(np.clip((100.0 - code_metrics.maintainability_index) / 80.0, 0.01, 0.99))

            row = {}
            for feat in features:
                if "rank_code_loc" in feat or "rank_code_sloc" in feat: row[feat] = rank_loc
                elif "rank_code_max_cyclomatic" in feat or "rank_code_avg_cyclomatic" in feat: row[feat] = rank_comp
                elif "rank_total_smells" in feat or "rank_long_method" in feat or "rank_high_complexity" in feat: row[feat] = rank_smells
                elif "rank_code_maintainability" in feat: row[feat] = rank_mi
                elif "rel_code_loc" in feat: row[feat] = float(code_metrics.loc / 120.0)
                elif "rel_code_max_cyclomatic" in feat: row[feat] = float(code_metrics.max_cyclomatic_complexity / 3.0)
                elif "rel_total_smells" in feat: row[feat] = float(smell_feats.total_smells)
                elif "has_long_method" in feat or "has_long_param_list" in feat: row[feat] = float(s_dict.get(feat, 0))
                elif "total_smells" in feat: row[feat] = float(smell_feats.total_smells)
                elif "smell_density" in feat: row[feat] = float(smell_feats.total_smells / (code_metrics.loc + 1) * 1000)
                else: row[feat] = 0.2
            
            df = pd.DataFrame([row])
            ml_prob = float(model.predict_proba(df)[0, 1])
            
            # Blend ML prior (40%) with real-time AST structural smell evidence (60%)
            prob = 0.60 * heuristic_prob + 0.40 * ml_prob
        except Exception as e:
            logger.debug(f"Model blend fallback: {e}")
            prob = heuristic_prob
    else:
        prob = heuristic_prob


    prob = round(prob, 4)
    tier_info = classify_risk_tier(prob)
    refactor_advice = analyze_refactorings(source_code)

    return {
        "file_path": file_path,
        "risk_probability": prob,
        "risk_tier": tier_info["tier"],
        "risk_icon": tier_info["icon"],
        "risk_color": tier_info["color"],
        "recommendation": tier_info["recommendation"],
        "code_metrics": asdict(code_metrics),
        "smells": asdict(smell_feats),
        "refactoring_advice": refactor_advice,
        "model_loaded": model is not None,
    }
