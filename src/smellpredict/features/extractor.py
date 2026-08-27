"""
SmellPredict — Feature Extraction Module
==========================================
Extracts all features from a Python source file snapshot:

  A. Traditional source-code metrics (15+)
  B. Code smell features (8+) with configurable thresholds
  C. Development-history metrics (14+)
  D. Structural / advanced metrics (Halstead, MI, Cognitive Complexity)

All extraction is AST-based using Python's built-in `ast` module + `radon`.
"""

from __future__ import annotations

import ast
import math
import re
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Optional

try:
    from radon.complexity import cc_visit
    from radon.metrics import h_visit, mi_visit
    from radon.raw import analyze
    HAS_RADON = True
except ImportError:
    HAS_RADON = False


# ─────────────────────────────────────────────────────────────────────────────
# Default Smell Thresholds (paper values)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_LONG_METHOD = 50
THRESHOLD_LONG_PARAM = 5
THRESHOLD_LARGE_CLASS = 300
THRESHOLD_DEEP_NESTING = 4
THRESHOLD_HIGH_COMPLEXITY = 10


# ─────────────────────────────────────────────────────────────────────────────
# Dataclasses for typed feature output
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeMetrics:
    """Traditional source-code metrics (Feature Group A)."""
    loc: int = 0                        # Lines of code
    sloc: int = 0                       # Source lines (non-blank, non-comment)
    blank_lines: int = 0
    comment_lines: int = 0
    comment_density: float = 0.0        # comment_lines / max(loc, 1)
    function_count: int = 0
    class_count: int = 0
    import_count: int = 0
    avg_function_size: float = 0.0
    max_function_size: int = 0
    avg_param_count: float = 0.0
    max_param_count: int = 0
    max_nesting_depth: int = 0
    avg_cyclomatic_complexity: float = 0.0
    max_cyclomatic_complexity: int = 0
    # Halstead metrics (radon)
    halstead_volume: float = 0.0
    halstead_difficulty: float = 0.0
    halstead_effort: float = 0.0
    halstead_bugs: float = 0.0          # estimated bugs = volume / 3000
    maintainability_index: float = 0.0  # 0–100, higher is more maintainable
    cognitive_complexity: int = 0       # custom AST walker


@dataclass
class SmellFeatures:
    """Code smell indicators (Feature Group B)."""
    # Binary indicators (1 if smell detected, else 0)
    has_long_method: int = 0
    has_long_param_list: int = 0
    has_large_class: int = 0
    has_deep_nesting: int = 0
    has_high_complexity: int = 0
    # Count features (how many instances)
    long_method_count: int = 0
    long_param_count: int = 0
    large_class_count: int = 0
    deep_nesting_count: int = 0
    high_complexity_count: int = 0
    # Aggregate
    total_smells: int = 0


@dataclass
class HistoryMetrics:
    """Development-history metrics (Feature Group C) — filled by mining module."""
    previous_file_commits: int = 0
    previous_bug_fixes: int = 0
    contributors: int = 0
    recent_file_commits: int = 0          # commits in last 30 days
    code_churn_history: int = 0           # total lines added+deleted
    file_age_days: float = 0.0
    days_since_last_change: float = 0.0
    developer_experience: int = 0         # author's total commits to project
    ownership_concentration: float = 0.0  # % commits by top contributor
    commit_message_entropy: float = 0.0   # information diversity
    avg_commit_size: float = 0.0          # avg lines changed per commit
    avg_time_between_commits: float = 0.0 # days
    has_multiple_contributors: int = 0    # binary flag
    is_recently_touched: int = 0          # changed in last 7 days


@dataclass
class FileFeatures:
    """Full feature vector for a single file snapshot."""
    # Identifiers
    repo: str = ""
    file_path: str = ""
    canonical_file_id: str = ""
    snapshot_commit: str = ""
    snapshot_date: str = ""
    # Feature groups
    code: CodeMetrics = field(default_factory=CodeMetrics)
    smells: SmellFeatures = field(default_factory=SmellFeatures)
    history: HistoryMetrics = field(default_factory=HistoryMetrics)
    # Label
    future_bug_fix: Optional[int] = None
    future_bug_fix_score: Optional[float] = None

    def to_flat_dict(self) -> dict:
        """Flatten all nested dataclasses to a single-level dict."""
        d = {}
        d["repo"] = self.repo
        d["file_path"] = self.file_path
        d["canonical_file_id"] = self.canonical_file_id
        d["snapshot_commit"] = self.snapshot_commit
        d["snapshot_date"] = self.snapshot_date
        for k, v in asdict(self.code).items():
            d[f"code_{k}"] = v
        for k, v in asdict(self.smells).items():
            d[k] = v
        for k, v in asdict(self.history).items():
            d[k] = v
        if self.future_bug_fix is not None:
            d["future_bug_fix"] = self.future_bug_fix
            d["future_bug_fix_score"] = self.future_bug_fix_score
        return d


# ─────────────────────────────────────────────────────────────────────────────
# AST Walkers
# ─────────────────────────────────────────────────────────────────────────────

class NestingDepthVisitor(ast.NodeVisitor):
    """Walk AST to find maximum nesting depth of control flow."""

    def __init__(self):
        self.max_depth = 0
        self._current_depth = 0

    def _enter_block(self, node):
        self._current_depth += 1
        self.max_depth = max(self.max_depth, self._current_depth)
        self.generic_visit(node)
        self._current_depth -= 1

    visit_If = visit_For = visit_While = visit_With = \
        visit_Try = visit_ExceptHandler = _enter_block


class CognitiveComplexityVisitor(ast.NodeVisitor):
    """
    Approximate cognitive complexity (Sonar-style).
    Increments for: if/elif/else, for, while, try/except,
    boolean operators (and/or), nested functions.
    Nesting level multiplier applied to structural increments.
    """

    def __init__(self):
        self.score = 0
        self._nesting = 0

    def _structural(self, node):
        self.score += 1 + self._nesting
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    visit_If = visit_For = visit_While = _structural

    def visit_Try(self, node):
        self.score += 1 + self._nesting
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    def visit_BoolOp(self, node):
        # Each and/or adds 1
        self.score += 1
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        # Nested function definition
        if self._nesting > 0:
            self.score += self._nesting
        self._nesting += 1
        self.generic_visit(node)
        self._nesting -= 1

    visit_AsyncFunctionDef = visit_FunctionDef


def _get_nesting_depth(tree: ast.AST) -> int:
    visitor = NestingDepthVisitor()
    visitor.visit(tree)
    return visitor.max_depth


def _get_cognitive_complexity(tree: ast.AST) -> int:
    visitor = CognitiveComplexityVisitor()
    visitor.visit(tree)
    return visitor.score


# ─────────────────────────────────────────────────────────────────────────────
# Core Extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_code_metrics(source: str) -> CodeMetrics:
    """
    Extract all source-code metrics from a Python file's source string.

    Args:
        source: Raw Python source code as a string

    Returns:
        CodeMetrics dataclass (all zeros if parsing fails)
    """
    metrics = CodeMetrics()

    if not source or not source.strip():
        return metrics

    # ── Raw line counts ──────────────────────────────────────────────────────
    lines = source.splitlines()
    metrics.loc = len(lines)
    metrics.comment_lines = sum(1 for l in lines if l.strip().startswith("#"))
    metrics.blank_lines = sum(1 for l in lines if not l.strip())
    metrics.sloc = metrics.loc - metrics.comment_lines - metrics.blank_lines
    metrics.comment_density = metrics.comment_lines / max(metrics.loc, 1)

    # ── Radon raw + Halstead + MI ────────────────────────────────────────────
    if HAS_RADON:
        try:
            raw = analyze(source)
            metrics.loc = raw.loc
            metrics.sloc = raw.lloc
            metrics.blank_lines = raw.blank
            metrics.comment_lines = raw.comments
            metrics.comment_density = raw.comments / max(raw.loc, 1)
        except Exception:
            pass

        try:
            h = h_visit(source)
            if h:
                hh = h[0]  # module-level Halstead
                metrics.halstead_volume = round(hh.volume, 2)
                metrics.halstead_difficulty = round(hh.difficulty, 2)
                metrics.halstead_effort = round(hh.effort, 2)
                metrics.halstead_bugs = round(hh.bugs, 4)
        except Exception:
            pass

        try:
            mi = mi_visit(source, multi=False)
            metrics.maintainability_index = round(mi, 2) if mi is not None else 0.0
        except Exception:
            pass

    # ── AST-based metrics ────────────────────────────────────────────────────
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return metrics  # return what we have from radon

    # Imports
    metrics.import_count = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
    )

    # Classes
    class_nodes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    metrics.class_count = len(class_nodes)

    # Functions (top-level and methods)
    func_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    metrics.function_count = len(func_nodes)

    if func_nodes:
        func_sizes = []
        param_counts = []
        for fn in func_nodes:
            # Size = last line - first line + 1
            try:
                size = fn.end_lineno - fn.lineno + 1
            except AttributeError:
                size = 1
            func_sizes.append(size)
            params = len(fn.args.args) + len(fn.args.posonlyargs)
            param_counts.append(params)

        metrics.avg_function_size = round(sum(func_sizes) / len(func_sizes), 2)
        metrics.max_function_size = max(func_sizes)
        metrics.avg_param_count = round(sum(param_counts) / len(param_counts), 2)
        metrics.max_param_count = max(param_counts)

    # Nesting depth
    metrics.max_nesting_depth = _get_nesting_depth(tree)

    # Cyclomatic complexity (via radon)
    if HAS_RADON:
        try:
            cc_results = cc_visit(source)
            if cc_results:
                complexities = [r.complexity for r in cc_results]
                metrics.avg_cyclomatic_complexity = round(
                    sum(complexities) / len(complexities), 2
                )
                metrics.max_cyclomatic_complexity = max(complexities)
        except Exception:
            pass

    # Cognitive complexity
    metrics.cognitive_complexity = _get_cognitive_complexity(tree)

    return metrics


def extract_smell_features(
    source: str,
    thresholds: Optional[dict] = None,
) -> SmellFeatures:
    """
    Extract code smell features using rule-based threshold detection.

    Args:
        source: Python source code string
        thresholds: Optional dict overriding default thresholds.
                    Keys: long_method, long_param, large_class,
                          deep_nesting, high_complexity

    Returns:
        SmellFeatures dataclass
    """
    t = {
        "long_method": THRESHOLD_LONG_METHOD,
        "long_param": THRESHOLD_LONG_PARAM,
        "large_class": THRESHOLD_LARGE_CLASS,
        "deep_nesting": THRESHOLD_DEEP_NESTING,
        "high_complexity": THRESHOLD_HIGH_COMPLEXITY,
    }
    if thresholds:
        t.update(thresholds)

    smells = SmellFeatures()

    if not source or not source.strip():
        return smells

    try:
        tree = ast.parse(source)
    except SyntaxError:
        return smells

    # ── Long Method ──────────────────────────────────────────────────────────
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                size = fn.end_lineno - fn.lineno + 1
            except AttributeError:
                size = 0
            if size >= t["long_method"]:
                smells.long_method_count += 1

    # ── Long Parameter List ───────────────────────────────────────────────────
    for fn in ast.walk(tree):
        if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            n_params = len(fn.args.args) + len(fn.args.posonlyargs)
            if n_params >= t["long_param"]:
                smells.long_param_count += 1

    # ── Large Class ───────────────────────────────────────────────────────────
    for cls in ast.walk(tree):
        if isinstance(cls, ast.ClassDef):
            try:
                size = cls.end_lineno - cls.lineno + 1
            except AttributeError:
                size = 0
            if size >= t["large_class"]:
                smells.large_class_count += 1

    # ── Deep Nesting ─────────────────────────────────────────────────────────
    max_nest = _get_nesting_depth(tree)
    if max_nest >= t["deep_nesting"]:
        smells.deep_nesting_count = 1  # file-level: 0 or 1

    # ── High Cyclomatic Complexity ────────────────────────────────────────────
    if HAS_RADON:
        try:
            cc_results = cc_visit(source)
            for r in cc_results:
                if r.complexity >= t["high_complexity"]:
                    smells.high_complexity_count += 1
        except Exception:
            pass

    # ── Binary indicators ─────────────────────────────────────────────────────
    smells.has_long_method = int(smells.long_method_count > 0)
    smells.has_long_param_list = int(smells.long_param_count > 0)
    smells.has_large_class = int(smells.large_class_count > 0)
    smells.has_deep_nesting = int(smells.deep_nesting_count > 0)
    smells.has_high_complexity = int(smells.high_complexity_count > 0)

    smells.total_smells = (
        smells.long_method_count
        + smells.long_param_count
        + smells.large_class_count
        + smells.deep_nesting_count
        + smells.high_complexity_count
    )

    return smells


def extract_file_features(
    source: str,
    thresholds: Optional[dict] = None,
) -> tuple[CodeMetrics, SmellFeatures]:
    """
    Convenience function: extract code metrics + smell features from source.

    Returns:
        (CodeMetrics, SmellFeatures) tuple
    """
    code = extract_code_metrics(source)
    smells = extract_smell_features(source, thresholds)
    return code, smells


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_code = '''
def very_long_function(param_a, param_b, param_c, param_d, param_e, param_f):
    """A function with many parameters and deep nesting."""
    if param_a:
        for i in range(100):
            while param_b:
                if param_c:
                    try:
                        result = param_d + param_e
                    except Exception:
                        pass
    return None

class BigClass:
    """A large class."""
    pass
'''
    code, smells = extract_file_features(sample_code)
    print("Code Metrics:")
    print(f"  LOC={code.loc}, Functions={code.function_count}, "
          f"MaxNesting={code.max_nesting_depth}, MaxParams={code.max_param_count}")
    print("Smell Features:")
    print(f"  LongMethod={smells.has_long_method}, LongParam={smells.has_long_param_list}, "
          f"DeepNesting={smells.has_deep_nesting}, TotalSmells={smells.total_smells}")
