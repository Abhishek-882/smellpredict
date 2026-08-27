"""
SmellPredict — SARIF (Static Analysis Results Interchange Format) Exporter
==========================================================================
Generates OASIS SARIF v2.1.0 JSON reports for IDE integrations (VS Code,
JetBrains) and GitHub Security Code Scanning / CodeQL alerts.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

SARIF_RULES = [
    {
        "id": "SP-RISK-001",
        "name": "CriticalDefectRisk",
        "shortDescription": {"text": "Critical bug-fix defect risk predicted by ML model"},
        "fullDescription": {"text": "The calibrated Random Forest model predicted a defect risk probability >= 80% based on complexity, maintainability, and code smells."},
        "defaultConfiguration": {"level": "error"},
        "properties": {"tags": ["defect-risk", "ml-prediction", "quality"]},
    },
    {
        "id": "SP-RISK-002",
        "name": "HighDefectRisk",
        "shortDescription": {"text": "High bug-fix defect risk predicted by ML model"},
        "fullDescription": {"text": "The calibrated Random Forest model predicted a defect risk probability between 65% and 80%."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["defect-risk", "ml-prediction", "quality"]},
    },
    {
        "id": "SP-SMELL-001",
        "name": "LongMethod",
        "shortDescription": {"text": "Long Method code smell detected (LOC >= 50)"},
        "fullDescription": {"text": "Methods with excessive lines of code are harder to read, maintain, and test."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["code-smell", "maintainability"]},
    },
    {
        "id": "SP-SMELL-002",
        "name": "LongParameterList",
        "shortDescription": {"text": "Long Parameter List code smell detected (params >= 5)"},
        "fullDescription": {"text": "Functions with many parameters increase cognitive load and indicate missing parameter object abstractions."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["code-smell", "design"]},
    },
    {
        "id": "SP-SMELL-003",
        "name": "LargeClass",
        "shortDescription": {"text": "Large Class / God Class code smell detected (LOC >= 300)"},
        "fullDescription": {"text": "Classes with hundreds of lines violate the Single Responsibility Principle."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["code-smell", "architecture"]},
    },
    {
        "id": "SP-SMELL-004",
        "name": "DeepNesting",
        "shortDescription": {"text": "Deep Nesting code smell detected (depth >= 4)"},
        "fullDescription": {"text": "Deeply nested control structures significantly increase cyclomatic complexity and defect susceptibility."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["code-smell", "complexity"]},
    },
    {
        "id": "SP-SMELL-005",
        "name": "HighComplexity",
        "shortDescription": {"text": "High Cyclomatic Complexity code smell detected (CC >= 10)"},
        "fullDescription": {"text": "Complex branching logic exponentially multiplies the required test paths."},
        "defaultConfiguration": {"level": "warning"},
        "properties": {"tags": ["code-smell", "complexity"]},
    },
]


def generate_sarif_report(
    file_results: List[Dict[str, Any]],
    tool_version: str = "2.0.0",
) -> Dict[str, Any]:
    """
    Convert a list of file analysis dicts into a standard SARIF v2.1.0 structure.
    
    file_results structure expected:
    [
        {
            "file_path": "src/module.py",
            "risk_probability": 0.85,
            "risk_tier": "Critical",
            "smells": {
                "has_long_method": 1,
                "has_long_param_list": 0,
                "has_large_class": 1,
                "has_deep_nesting": 1,
                "has_high_complexity": 1
            },
            "metrics": {
                "loc": 350,
                "max_cc": 14.0,
                "maintainability_index": 45.2
            }
        }, ...
    ]
    """
    sarif_results = []

    for item in file_results:
        fpath = str(item.get("file_path", "unknown.py")).replace("\\", "/")
        risk_prob = item.get("risk_probability", 0.0)
        smells = item.get("smells", {})
        metrics = item.get("metrics", {})

        # 1. Defect Risk Level Diagnostic
        if risk_prob >= 0.80:
            sarif_results.append({
                "ruleId": "SP-RISK-001",
                "level": "error",
                "message": {
                    "text": (
                        f"Critical defect risk predicted: {risk_prob*100:.1f}%. "
                        f"Maintainability: {metrics.get('maintainability_index', 0):.1f}/100, "
                        f"Max CC: {metrics.get('max_cc', 0):.1f}."
                    )
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": fpath, "uriBaseId": "%SRCROOT%"},
                        "region": {"startLine": 1, "startColumn": 1}
                    }
                }]
            })
        elif risk_prob >= 0.65:
            sarif_results.append({
                "ruleId": "SP-RISK-002",
                "level": "warning",
                "message": {
                    "text": (
                        f"High defect risk predicted: {risk_prob*100:.1f}%. "
                        f"Consider refactoring detected code smells."
                    )
                },
                "locations": [{
                    "physicalLocation": {
                        "artifactLocation": {"uri": fpath, "uriBaseId": "%SRCROOT%"},
                        "region": {"startLine": 1, "startColumn": 1}
                    }
                }]
            })

        # 2. Individual Smell Diagnostics
        smell_map = [
            ("has_long_method", "SP-SMELL-001", "Long Method smell detected (LOC exceeds threshold)."),
            ("has_long_param_list", "SP-SMELL-002", "Long Parameter List detected (>= 5 parameters)."),
            ("has_large_class", "SP-SMELL-003", "Large Class smell detected (Class LOC >= 300)."),
            ("has_deep_nesting", "SP-SMELL-004", "Deep Nesting detected (Control flow depth >= 4)."),
            ("has_high_complexity", "SP-SMELL-005", "High Cyclomatic Complexity detected (CC >= 10)."),
        ]

        for smell_key, rule_id, desc in smell_map:
            if smells.get(smell_key, 0) == 1:
                sarif_results.append({
                    "ruleId": rule_id,
                    "level": "warning",
                    "message": {"text": desc},
                    "locations": [{
                        "physicalLocation": {
                            "artifactLocation": {"uri": fpath, "uriBaseId": "%SRCROOT%"},
                            "region": {"startLine": 1, "startColumn": 1}
                        }
                    }]
                })

    return {
        "$schema": "https://raw.githubusercontent.com/oasis-tcs/sarif-spec/master/Schemata/sarif-schema-2.1.0.json",
        "version": "2.1.0",
        "runs": [{
            "tool": {
                "driver": {
                    "name": "SmellPredict",
                    "version": tool_version,
                    "informationUri": "https://github.com/smellpredict/smellpredict",
                    "rules": SARIF_RULES
                }
            },
            "results": sarif_results
        }]
    }


def export_sarif_file(file_results: List[Dict[str, Any]], output_path: Path) -> Path:
    """Generate and write SARIF report to a file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sarif_data = generate_sarif_report(file_results)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sarif_data, f, indent=2)
    return output_path
