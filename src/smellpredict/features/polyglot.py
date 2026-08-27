"""
SmellPredict — Multi-Language Polyglot Code Smell Analyzer
==========================================================
Provides universal source code metric extraction and heuristic code smell
detection across enterprise programming languages:
- Python (.py)
- Java (.java)
- Kotlin (.kt, .kts)
- JavaScript & TypeScript (.js, .jsx, .ts, .tsx, .mjs, .cjs)
- Go (.go)
- Rust (.rs)
- C / C++ (.c, .h, .cpp, .hpp, .cc)
- Ruby (.rb)
- PHP (.php)
- Swift (.swift)
- Markdown & Config (.md, .json, .yaml, .yml, .toml)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class PolyglotMetrics:
    """Universal code metrics for any programming language."""
    language: str = "plaintext"
    loc: int = 0
    sloc: int = 0
    blank_lines: int = 0
    comment_lines: int = 0
    comment_density: float = 0.0
    function_count: int = 0
    class_count: int = 0
    max_function_size: int = 0
    avg_function_size: float = 0.0
    max_param_count: int = 0
    avg_param_count: float = 0.0
    max_nesting_depth: int = 0
    cyclomatic_complexity: int = 0
    todo_count: int = 0


@dataclass
class PolyglotSmells:
    """Code smell detection flags and counts."""
    has_long_method: int = 0
    has_long_param_list: int = 0
    has_large_class: int = 0
    has_deep_nesting: int = 0
    has_high_complexity: int = 0
    has_god_file: int = 0
    total_smells: int = 0
    smell_details: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class PolyglotAnalysisResult:
    """Complete analysis output for IDE and API."""
    language: str
    file_path: str
    risk_probability: float
    risk_tier: str
    risk_icon: str
    recommendation: str
    metrics: PolyglotMetrics
    smells: PolyglotSmells
    refactoring_advice: List[Dict[str, Any]] = field(default_factory=list)
    is_ml_prediction: bool = False
    model_name: str = "Static Telemetry (Untrained ML)"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "filename": Path(self.file_path).name,
            "file_path": self.file_path,
            "is_ml_prediction": False,
            "model_name": self.model_name,
            "risk": None,
            "engine_used": "none",
            "engine_desc": "No ML model for this language (Python Only)",
            "guardrail_status": "unsupported_language",
            "confidence_warning": f"ML defect risk model is strictly trained on Python codebases. No risk evaluation is performed for {self.language.capitalize()}.",
            "metrics": asdict(self.metrics),
            "smells": {
                "has_long_method": self.smells.has_long_method,
                "has_long_param_list": self.smells.has_long_param_list,
                "has_large_class": self.smells.has_large_class,
                "has_deep_nesting": self.smells.has_deep_nesting,
                "has_high_complexity": self.smells.has_high_complexity,
                "total_smells": self.smells.total_smells,
                "details": self.smells.smell_details,
            },
            "refactoring": self.refactoring_advice,
        }


# ── Language Detection Map ───────────────────────────────────────────────────

EXTENSION_MAP: Dict[str, str] = {
    ".py": "python",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cc": "cpp",
    ".cxx": "cpp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".md": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".sh": "shell",
    ".bash": "shell",
}

LANGUAGE_BADGES: Dict[str, str] = {
    "python": "🐍 Python",
    "java": "☕ Java",
    "kotlin": "🟣 Kotlin",
    "javascript": "🟨 JavaScript",
    "typescript": "🔷 TypeScript",
    "go": "🐹 Go",
    "rust": "🦀 Rust",
    "c": "⚙️ C",
    "cpp": "⚙️ C++",
    "ruby": "💎 Ruby",
    "php": "🐘 PHP",
    "swift": "🐦 Swift",
    "markdown": "📝 Markdown",
    "json": "📋 JSON",
    "yaml": "⚙️ YAML",
    "shell": "💻 Shell",
    "plaintext": "📄 Text",
}


# ── Language Patterns Configuration ──────────────────────────────────────────

LANG_CONFIGS: Dict[str, Dict[str, Any]] = {
    "java": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:public|protected|private|static|\s)+[\w\<\>\[\]]+\s+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?:throws\s+[\w,\s]+)?\s*\{",
        "class_pattern": r"(?:public|protected|private|\s)*(?:class|interface|enum|record)\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\bcatch\b", r"\b&&|\b\|\|", r"\bswitch\b"],
    },
    "kotlin": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:fun|override\s+fun|suspend\s+fun)\s+(?:<[^>]+>\s+)?([a-zA-Z_]\w*)\s*\(([^)]*)\)",
        "class_pattern": r"(?:class|interface|object|enum\s+class|data\s+class|sealed\s+class)\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bwhen\b", r"\bcatch\b", r"\b&&|\b\|\|"],
    },
    "javascript": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:async\s+)?function\s*([a-zA-Z_]\w*)?\s*\(([^)]*)\)|(?:const|let|var)\s+([a-zA-Z_]\w*)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>|([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{",
        "class_pattern": r"class\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\bcatch\b", r"\b&&|\b\|\|", r"\bswitch\b"],
    },
    "typescript": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:async\s+)?(?:public|private|protected|static\s+)?function\s*([a-zA-Z_]\w*)?\s*\(([^)]*)\)|(?:const|let|var)\s+([a-zA-Z_]\w*)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*(?::\s*[^=]+)?\s*=>|([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?::\s*[^\{]+)?\s*\{",
        "class_pattern": r"(?:class|interface|type|enum)\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\bcatch\b", r"\b&&|\b\|\|", r"\bswitch\b"],
    },
    "go": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"func\s*(?:\([^)]+\)\s*)?([a-zA-Z_]\w*)\s*\(([^)]*)\)",
        "class_pattern": r"type\s+([a-zA-Z_]\w*)\s+struct",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bcase\b", r"\bselect\b", r"\b&&|\b\|\|"],
    },
    "rust": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:pub\s+)?(?:async\s+)?fn\s+([a-zA-Z_]\w*)\s*(?:<[^>]+>)?\s*\(([^)]*)\)",
        "class_pattern": r"(?:pub\s+)?(?:struct|enum|trait|impl(?:\s*<[^>]+>)?)\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bmatch\b", r"\bloop\b", r"\b&&|\b\|\|"],
    },
    "cpp": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:[\w\<\>:]+\s+)+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*(?:const)?\s*\{",
        "class_pattern": r"(?:class|struct)\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\bcatch\b", r"\b&&|\b\|\|", r"\bswitch\b"],
    },
    "c": {
        "comment_line": r"^\s*//",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "func_pattern": r"(?:[\w\*]+\s+)+([a-zA-Z_]\w*)\s*\(([^)]*)\)\s*\{",
        "class_pattern": r"struct\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belse\b", r"\bfor\b", r"\bwhile\b", r"\bcase\b", r"\b&&|\b\|\|", r"\bswitch\b"],
    },
    "ruby": {
        "comment_line": r"^\s*#",
        "comment_block_start": r"=begin",
        "comment_block_end": r"=end",
        "func_pattern": r"def\s+([a-zA-Z_]\w*[!?=]?)(?:\s*\(([^)]*)\))?",
        "class_pattern": r"(?:class|module)\s+([a-zA-Z_]\w*)",
        "branch_keywords": [r"\bif\b", r"\belsif\b", r"\belse\b", r"\bunless\b", r"\bwhile\b", r"\buntil\b", r"\bfor\b", r"\bcase\b", r"\brescue\b"],
    },
}


# ── Universal Metric Extractor ───────────────────────────────────────────────

def extract_polyglot_metrics(source: str, language: str) -> PolyglotMetrics:
    """Extract line counts, functions, nesting, and complexity for any language."""
    metrics = PolyglotMetrics(language=language)
    if not source or not source.strip():
        return metrics

    lines = source.splitlines()
    metrics.loc = len(lines)

    config = LANG_CONFIGS.get(language, {
        "comment_line": r"^\s*(?://|#)",
        "comment_block_start": r"/\*",
        "comment_block_end": r"\*/",
        "branch_keywords": [r"\bif\b", r"\bfor\b", r"\bwhile\b"],
    })

    comment_line_re = re.compile(config.get("comment_line", r"^\s*//"))
    block_start_re = re.compile(config.get("comment_block_start", r"/\*"))
    block_end_re = re.compile(config.get("comment_block_end", r"\*/"))

    in_block_comment = False
    clean_lines: List[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            metrics.blank_lines += 1
            continue

        if in_block_comment:
            metrics.comment_lines += 1
            if block_end_re.search(line):
                in_block_comment = False
            continue

        if block_start_re.search(line):
            metrics.comment_lines += 1
            if not block_end_re.search(line):
                in_block_comment = True
            continue

        if comment_line_re.match(line):
            metrics.comment_lines += 1
            if "TODO" in line or "FIXME" in line or "HACK" in line:
                metrics.todo_count += 1
            continue

        if "TODO" in line or "FIXME" in line or "HACK" in line:
            metrics.todo_count += 1

        clean_lines.append(line)

    metrics.sloc = max(0, metrics.loc - metrics.comment_lines - metrics.blank_lines)
    metrics.comment_density = round(metrics.comment_lines / max(metrics.loc, 1), 3)

    # Nesting depth analysis
    max_depth = 0
    current_depth = 0
    for line in lines:
        open_braces = line.count("{")
        close_braces = line.count("}")
        current_depth = max(0, current_depth + open_braces - close_braces)
        if current_depth > max_depth:
            max_depth = current_depth

        # Indentation based fallback (e.g. Python, Ruby)
        if open_braces == 0 and close_braces == 0 and line.strip():
            indent = len(line) - len(line.lstrip())
            indent_depth = indent // 4
            if indent_depth > max_depth:
                max_depth = indent_depth

    metrics.max_nesting_depth = max_depth

    # Branching & Complexity estimation
    complexity = 1
    branch_regexes = [re.compile(kw) for kw in config.get("branch_keywords", [])]
    for line in clean_lines:
        for r in branch_regexes:
            complexity += len(r.findall(line))
    metrics.cyclomatic_complexity = complexity

    # Function & Class parsing
    func_pattern = config.get("func_pattern")
    class_pattern = config.get("class_pattern")

    if func_pattern:
        func_re = re.compile(func_pattern)
        funcs = func_re.findall(source)
        metrics.function_count = len(funcs)

        param_counts = []
        for match in funcs:
            params_str = ""
            if isinstance(match, tuple):
                for elem in match:
                    if "," in elem or (elem and not elem.isidentifier()):
                        params_str = elem
                        break
            elif isinstance(match, str):
                params_str = match

            if params_str and params_str.strip():
                # Count comma-separated arguments
                params = [p.strip() for p in params_str.split(",") if p.strip()]
                param_counts.append(len(params))
            else:
                param_counts.append(0)

        if param_counts:
            metrics.max_param_count = max(param_counts)
            metrics.avg_param_count = round(sum(param_counts) / len(param_counts), 1)

        if metrics.function_count > 0:
            metrics.avg_function_size = round(metrics.sloc / metrics.function_count, 1)
            metrics.max_function_size = int(metrics.avg_function_size * 1.6)

    if class_pattern:
        class_re = re.compile(class_pattern)
        classes = class_re.findall(source)
        metrics.class_count = len(classes)

    return metrics


# ── Smell Detection & Risk Scoring ───────────────────────────────────────────

def detect_polyglot_smells(metrics: PolyglotMetrics) -> Tuple[PolyglotSmells, float, str, str, str, List[Dict[str, Any]]]:
    """Detect code smells and compute risk probability + tier + recommendations."""
    smells = PolyglotSmells()
    details = []
    advice = []
    risk_score = 0.0

    # 1. Long Method (> 35 SLOC per function or max func > 45)
    if metrics.avg_function_size > 35 or metrics.max_function_size > 45:
        smells.has_long_method = 1
        smells.total_smells += 1
        risk_score += 0.22
        details.append({
            "smell": "Long Method",
            "severity": "Medium",
            "message": f"Average function size is {metrics.avg_function_size} lines (threshold: 35).",
        })
        advice.append({
            "title": "Extract Method",
            "smell_type": "Long Method",
            "suggested_action": "Decompose long routines into modular, single-responsibility sub-functions.",
            "description": "Functions exceeding 35 lines increase defect density and testing complexity.",
        })

    # 2. Long Parameter List (> 4 params)
    if metrics.max_param_count >= 5:
        smells.has_long_param_list = 1
        smells.total_smells += 1
        risk_score += 0.18
        details.append({
            "smell": "Long Parameter List",
            "severity": "Medium",
            "message": f"Method contains {metrics.max_param_count} parameters (threshold: 4).",
        })
        advice.append({
            "title": "Introduce Parameter Object",
            "smell_type": "Long Parameter List",
            "suggested_action": "Encapsulate multiple related arguments into a structured data object or DTO.",
            "description": "Passing 5+ arguments leads to positional bugs and high cognitive overhead.",
        })

    # 3. Large Class / God File (> 250 SLOC or single class > 200)
    if (metrics.class_count <= 2 and metrics.sloc > 250) or metrics.sloc > 400:
        smells.has_large_class = 1
        smells.has_god_file = 1
        smells.total_smells += 1
        risk_score += 0.25
        details.append({
            "smell": "Large Class / God Module",
            "severity": "High",
            "message": f"File contains {metrics.sloc} source lines with concentrated responsibilities.",
        })
        advice.append({
            "title": "Separate Concerns",
            "smell_type": "Large Class",
            "suggested_action": "Extract secondary responsibilities (I/O, formatting, validation) into helper classes.",
            "description": "Large monolithic units have high defect recurrence upon modification.",
        })

    # 4. Deep Nesting (> 3 nested levels)
    if metrics.max_nesting_depth >= 4:
        smells.has_deep_nesting = 1
        smells.total_smells += 1
        risk_score += 0.20
        details.append({
            "smell": "Deep Nesting",
            "severity": "High",
            "message": f"Nesting depth reached {metrics.max_nesting_depth} levels (threshold: 3).",
        })
        advice.append({
            "title": "Introduce Guard Clauses",
            "smell_type": "Deep Nesting",
            "suggested_action": "Replace nested if-statements with early returns and inverted checks.",
            "description": "Deeply nested code significantly increases cognitive complexity and edge-case bugs.",
        })

    # 5. High Cyclomatic Complexity (> 15)
    if metrics.cyclomatic_complexity >= 15:
        smells.has_high_complexity = 1
        smells.total_smells += 1
        risk_score += 0.15
        details.append({
            "smell": "High Complexity",
            "severity": "High",
            "message": f"Cyclomatic complexity is {metrics.cyclomatic_complexity} (threshold: 15).",
        })
        advice.append({
            "title": "Simplify Branching Logic",
            "smell_type": "High Complexity",
            "suggested_action": "Replace complex conditional logic with polymorphism or lookup tables.",
            "description": "High cyclomatic complexity requires exponential branch coverage in tests.",
        })

    # TODO debt factor
    if metrics.todo_count >= 4:
        risk_score += min(0.10, metrics.todo_count * 0.02)
        details.append({
            "smell": "Technical Debt Comments",
            "severity": "Low",
            "message": f"Detected {metrics.todo_count} TODO/FIXME markers in source.",
        })

    smells.smell_details = details

    # Calibrate probability [0.05, 0.95]
    risk_probability = max(0.05, min(0.95, round(risk_score, 3)))

    if risk_probability < 0.20:
        risk_tier = "Low"
        risk_icon = "🟢"
        recommendation = "Clean code structure. Standard code review sufficient."
    elif risk_probability < 0.50:
        risk_tier = "Medium"
        risk_icon = "🟡"
        recommendation = "Moderate smell indicators. Consider applying quick-fix refactorings."
    elif risk_probability < 0.75:
        risk_tier = "High"
        risk_icon = "🟠"
        recommendation = "High defect probability. Recommended peer review & complexity reduction."
    else:
        risk_tier = "Critical"
        risk_icon = "🔴"
        recommendation = "Critical architectural debt. Mandatory refactoring before deployment."

    return smells, risk_probability, risk_tier, risk_icon, recommendation, advice


# ── Public Analyzer Entry Point ──────────────────────────────────────────────

def polyglot_analyze(source: str, file_path: str = "untitled.txt") -> PolyglotAnalysisResult:
    """
    Main polyglot analysis function.
    Extracts static metrics and structural smells.
    Transparently marks non-Python/non-Java languages as Untrained for ML risk inference.
    """
    ext = Path(file_path).suffix.lower()
    language = EXTENSION_MAP.get(ext, "plaintext")

    metrics = extract_polyglot_metrics(source, language)
    smells, _, _, _, _, advice = detect_polyglot_smells(metrics)

    # Scientific honesty: Do not fake ML bug predictions for untrained languages
    rec = f"SmellPredict defect ML model is trained on Python & Java repositories. For {language.capitalize()}, static code complexity telemetry is displayed without ML defect inference."

    return PolyglotAnalysisResult(
        language=language,
        file_path=file_path,
        risk_probability=0.0,
        risk_tier="Untrained",
        risk_icon="⚪",
        recommendation=rec,
        metrics=metrics,
        smells=smells,
        refactoring_advice=advice,
        is_ml_prediction=False,
        model_name=f"{language.capitalize()} Static Telemetry (No ML Model)",
    )
