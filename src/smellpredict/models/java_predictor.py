"""
SmellPredict — Java Model Predictor Module
===========================================
Unified predictor for Java source files using the dedicated Java CatBoost/RandomForest
model trained on 50+ Java OSS repositories.

Integrates:
  - Java AST feature extraction (java_extractor)
  - Trained Java ML model inference
  - Deterministic heuristic risk fallback (when model not trained yet)
  - Refactoring advice generator for Java
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from smellpredict.features.java_extractor import extract_java_metrics, extract_java_smells

_JAVA_MODEL_CACHE: Dict[str, Any] = {}

FEATURE_COLUMNS = [
    "code_loc", "code_sloc", "code_comment_density",
    "code_function_count", "code_class_count", "code_import_count",
    "code_avg_function_size", "code_max_function_size",
    "code_avg_param_count", "code_max_param_count",
    "code_max_nesting_depth",
    "code_avg_cyclomatic_complexity", "code_max_cyclomatic_complexity",
    "code_halstead_volume", "code_halstead_difficulty", "code_halstead_effort",
    "code_halstead_bugs", "code_maintainability_index", "code_cognitive_complexity",
    "has_long_method", "has_long_param_list", "has_large_class",
    "has_deep_nesting", "has_high_complexity",
    "long_method_count", "long_param_count", "large_class_count",
    "deep_nesting_count", "high_complexity_count", "total_smells",
]


def classify_java_risk_tier(prob: float) -> Dict[str, str]:
    """Map calibrated Java probability to risk tier, icon, and recommendation."""
    if prob < 0.25:
        return {
            "tier": "Low",
            "icon": "🟢",
            "recommendation": "Clean Java structure. Low risk of future defect modifications.",
            "color": "#10b981",
        }
    elif prob < 0.50:
        return {
            "tier": "Medium",
            "icon": "🟡",
            "recommendation": "Moderate structural complexity. Consider reviewing method lengths and class responsibilities.",
            "color": "#f59e0b",
        }
    elif prob < 0.75:
        return {
            "tier": "High",
            "icon": "🟠",
            "recommendation": "High defect risk. Multiple Java code smells detected. Refactoring recommended.",
            "color": "#f97316",
        }
    else:
        return {
            "tier": "Critical",
            "icon": "🔴",
            "recommendation": "Critical Java defect risk. Decompose God classes and deeply nested methods immediately.",
            "color": "#ef4444",
        }


def get_java_trained_model() -> Optional[Any]:
    """Lazy-load the trained Java model from disk."""
    if "model" not in _JAVA_MODEL_CACHE:
        model_paths = [
            Path(os.environ.get("SMELLPREDICT_JAVA_MODEL_PATH", "models/java_best_model.pkl")),
            Path(__file__).parent.parent.parent.parent / "models" / "java_best_model.pkl",
            Path("models/java_best_model.pkl"),
        ]
        loaded = None
        for p in model_paths:
            if p.exists():
                try:
                    with open(p, "rb") as f:
                        loaded = pickle.load(f)
                    _JAVA_MODEL_CACHE["model"] = loaded
                    _JAVA_MODEL_CACHE["model_source"] = str(p)
                    logger.info(f"Loaded trained Java model from {p}")
                    break
                except Exception as e:
                    logger.warning(f"Failed to load Java model from {p}: {e}")

        if loaded is None:
            _JAVA_MODEL_CACHE["model"] = None
            _JAVA_MODEL_CACHE["model_source"] = "none"

    return _JAVA_MODEL_CACHE.get("model")


def compute_java_heuristic_risk(metrics: dict, smells: dict) -> float:
    """Deterministic fallback risk score for Java when no trained model is present."""
    complexity_score = min(metrics.get("max_cyclomatic_complexity", 0) / 20.0, 1.0)
    nesting_score = min(metrics.get("max_nesting_depth", 0) / 6.0, 1.0)
    smell_score = min(smells.get("total_smells", 0) / 5.0, 1.0)
    loc_score = min(metrics.get("loc", 0) / 800.0, 1.0)
    mi_score = max(0.0, (100.0 - metrics.get("maintainability_index", 50.0)) / 100.0)

    prob = (
        0.30 * smell_score
        + 0.25 * complexity_score
        + 0.20 * nesting_score
        + 0.15 * loc_score
        + 0.10 * mi_score
    )
    return round(min(max(float(prob), 0.05), 0.95), 4)


def analyze_java_source_code(source: str, file_path: str = "sample.java") -> Dict[str, Any]:
    """
    Complete end-to-end analysis of Java source code:
    1. Extracts Java AST metrics & smells
    2. Runs inference using Java CatBoost/RF model (or calibrated heuristic fallback)
    3. Synthesizes actionable refactoring advice
    """
    metrics = extract_java_metrics(source)
    smells = extract_java_smells(source, metrics)

    model = get_java_trained_model()
    is_ml = False

    if model is not None:
        try:
            feat_dict = {
                "code_loc": metrics.get("loc", 0),
                "code_sloc": metrics.get("sloc", 0),
                "code_comment_density": metrics.get("comment_density", 0.0),
                "code_function_count": metrics.get("function_count", 0),
                "code_class_count": metrics.get("class_count", 0),
                "code_import_count": metrics.get("import_count", 0),
                "code_avg_function_size": metrics.get("avg_function_size", 0.0),
                "code_max_function_size": metrics.get("max_function_size", 0),
                "code_avg_param_count": metrics.get("avg_param_count", 0.0),
                "code_max_param_count": metrics.get("max_param_count", 0),
                "code_max_nesting_depth": metrics.get("max_nesting_depth", 0),
                "code_avg_cyclomatic_complexity": metrics.get("avg_cyclomatic_complexity", 0.0),
                "code_max_cyclomatic_complexity": metrics.get("max_cyclomatic_complexity", 0),
                "code_halstead_volume": metrics.get("halstead_volume", 0.0),
                "code_halstead_difficulty": metrics.get("halstead_difficulty", 0.0),
                "code_halstead_effort": metrics.get("halstead_effort", 0.0),
                "code_halstead_bugs": metrics.get("halstead_bugs", 0.0),
                "code_maintainability_index": metrics.get("maintainability_index", 0.0),
                "code_cognitive_complexity": metrics.get("cognitive_complexity", 0),
                **{k: smells.get(k, 0) for k in [
                    "has_long_method", "has_long_param_list", "has_large_class",
                    "has_deep_nesting", "has_high_complexity",
                    "long_method_count", "long_param_count", "large_class_count",
                    "deep_nesting_count", "high_complexity_count", "total_smells",
                ]},
            }
            X_df = pd.DataFrame([feat_dict])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X_df)[0, 1])
            else:
                prob = float(model.predict(X_df)[0])
            is_ml = True
        except Exception as e:
            logger.warning(f"Java ML model inference failed: {e} — falling back to heuristic")
            prob = compute_java_heuristic_risk(metrics, smells)
    else:
        prob = compute_java_heuristic_risk(metrics, smells)

    tier_info = classify_java_risk_tier(prob)
    advice = generate_java_refactoring_advice(source, smells)

    return {
        "file_path": file_path,
        "language": "java",
        "language_badge": "☕ Java",
        "risk_probability": round(prob, 4),
        "risk_tier": tier_info["tier"],
        "risk_icon": tier_info["icon"],
        "recommendation": tier_info["recommendation"],
        "risk_color": tier_info["color"],
        "is_ml_prediction": is_ml,
        "code_metrics": metrics,
        "smells": smells,
        "refactoring_advice": advice,
    }


def generate_java_refactoring_advice(source: str, smells: dict) -> List[Dict[str, Any]]:
    """Synthesize actionable refactoring advice for Java source code."""
    advice = []
    if smells.get("has_long_method"):
        advice.append({
            "title": "Extract Helper Method",
            "smell_type": "Long Method",
            "line_number": 1,
            "severity": "high",
            "description": f"Found {smells.get('long_method_count', 1)} long Java method(s) exceeding 50 lines.",
            "suggested_action": "Decompose large methods into private static or instance helper methods.",
            "code_template": "// Extract block into:\nprivate void processSubTask(...) {\n    // Extracted logic\n}",
        })
    if smells.get("has_long_param_list"):
        advice.append({
            "title": "Introduce Parameter Object (DTO/Record)",
            "smell_type": "Long Parameter List",
            "line_number": 1,
            "severity": "medium",
            "description": f"Found {smells.get('long_param_count', 1)} method(s) with > 5 parameters.",
            "suggested_action": "Encapsulate multiple related parameters into a Java record or builder POJO.",
            "code_template": "public record RequestContext(String id, double amount, Map<String, Object> meta) {}",
        })
    if smells.get("has_deep_nesting"):
        advice.append({
            "title": "Introduce Guard Clauses",
            "smell_type": "Deep Nesting",
            "line_number": 1,
            "severity": "high",
            "description": "Deep nesting level exceeds 4 braces.",
            "suggested_action": "Invert nested if-conditions and return early at the top of the method.",
            "code_template": "if (condition == null) return;\n// proceed with main logic",
        })
    if smells.get("has_large_class"):
        advice.append({
            "title": "Extract Delegate / Service Class",
            "smell_type": "Large Class",
            "line_number": 1,
            "severity": "high",
            "description": "Class size exceeds 300 LOC (God Class smell).",
            "suggested_action": "Apply Single Responsibility Principle (SRP) to decompose into cohesive service delegates.",
            "code_template": "public class OrderValidationService { ... }",
        })
    return advice
