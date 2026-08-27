"""
SmellPredict — Java AST Feature Extractor
==========================================
Extracts the SAME feature schema as extractor.py but for Java source files.

Feature Groups (identical column names for trainer.py compatibility):
  A. Code Metrics   → CODE_METRIC_COLS  (19 features)
  B. Smell Features → SMELL_COLS        (11 features)

Parser Strategy:
  Primary  : javalang AST (pure-Python, Java 8–15)
  Fallback : regex-based approximation (Java 16+, parse errors)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, asdict
from typing import Optional

from loguru import logger

try:
    import javalang
    HAS_JAVALANG = True
except ImportError:
    HAS_JAVALANG = False
    logger.warning("javalang not installed — Java extraction will use regex fallback only. Run: pip install javalang")


# ─────────────────────────────────────────────────────────────────────────────
# Smell Thresholds (same values as Python extractor.py)
# ─────────────────────────────────────────────────────────────────────────────

THRESHOLD_LONG_METHOD    = 50
THRESHOLD_LONG_PARAM     = 5
THRESHOLD_LARGE_CLASS    = 300
THRESHOLD_DEEP_NESTING   = 4
THRESHOLD_HIGH_COMPLEXITY = 10


# ─────────────────────────────────────────────────────────────────────────────
# Regex helpers (used both in primary and fallback paths)
# ─────────────────────────────────────────────────────────────────────────────

_RE_SINGLE_COMMENT  = re.compile(r'//.*')
_RE_BLOCK_COMMENT   = re.compile(r'/\*.*?\*/', re.DOTALL)
_RE_STRING_LITERAL  = re.compile(r'"(?:[^"\\]|\\.)*"')
_RE_METHOD_SIG      = re.compile(
    r'(?:(?:public|private|protected|static|final|abstract|synchronized|native|default)\s+)*'
    r'[\w<>\[\],?]+\s+(\w+)\s*\(([\s\S]*?)\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
)
_RE_CLASS_DECL      = re.compile(
    r'(?:public|private|protected|abstract|final|static)?\s*(?:class|interface|enum|record)\s+\w+',
    re.MULTILINE
)
_RE_IMPORT          = re.compile(r'^\s*import\s+', re.MULTILINE)
# Cyclomatic complexity decision points
_RE_DECISIONS       = re.compile(
    r'\b(if|else\s+if|for|while|case|catch|&&|\|\|)\b'
)
_RE_OPERATORS       = re.compile(r'[+\-*/=<>!&|^~%?:]+')
_RE_IDENTIFIERS     = re.compile(r'\b[a-zA-Z_]\w*\b')


# ─────────────────────────────────────────────────────────────────────────────
# Utility: strip comments and string literals (for clean analysis)
# ─────────────────────────────────────────────────────────────────────────────

def _strip_comments(source: str) -> str:
    """Remove block comments and single-line comments from Java source."""
    s = _RE_BLOCK_COMMENT.sub(' ', source)
    s = _RE_SINGLE_COMMENT.sub('', s)
    return s


def _count_comment_lines(source: str) -> int:
    """Count lines that are part of block or single-line comments."""
    count = 0
    in_block = False
    for line in source.splitlines():
        stripped = line.strip()
        if in_block:
            count += 1
            if '*/' in line:
                in_block = False
        elif stripped.startswith('//'):
            count += 1
        elif '/*' in stripped:
            count += 1
            if '*/' not in stripped[stripped.index('/*') + 2:]:
                in_block = True
    return count


# ─────────────────────────────────────────────────────────────────────────────
# Nesting Depth Walker (regex-based, works for both AST and fallback paths)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_max_nesting(source: str) -> int:
    """
    Compute max nesting depth by tracking { } pairs and
    nesting-increase keywords (if/for/while/try/switch/synchronized).
    """
    clean = _strip_comments(source)
    clean = _RE_STRING_LITERAL.sub('""', clean)

    depth = 0
    max_depth = 0
    for char in clean:
        if char == '{':
            depth += 1
            max_depth = max(max_depth, depth)
        elif char == '}':
            depth = max(0, depth - 1)
    # Approximate: deduct 1 for class-level scope (not real nesting)
    return max(0, max_depth - 1)


# ─────────────────────────────────────────────────────────────────────────────
# Cyclomatic Complexity (per method, regex-based)
# ─────────────────────────────────────────────────────────────────────────────

def _cyclomatic_complexity(method_body: str) -> int:
    """Compute McCabe CC = 1 + count of decision points in method body."""
    decisions = len(_RE_DECISIONS.findall(method_body))
    return 1 + decisions


# ─────────────────────────────────────────────────────────────────────────────
# Halstead Metrics (regex-based, language-agnostic)
# ─────────────────────────────────────────────────────────────────────────────

def _halstead_metrics(source: str) -> dict:
    """Compute Halstead volume, difficulty, effort, bugs from operator/operand counts."""
    clean = _strip_comments(source)
    clean = _RE_STRING_LITERAL.sub('""', clean)

    operators = _RE_OPERATORS.findall(clean)
    identifiers = _RE_IDENTIFIERS.findall(clean)

    # Filter Java keywords from identifiers
    _JAVA_KEYWORDS = frozenset([
        'abstract','assert','boolean','break','byte','case','catch','char','class',
        'const','continue','default','do','double','else','enum','extends','final',
        'finally','float','for','goto','if','implements','import','instanceof','int',
        'interface','long','native','new','null','package','private','protected',
        'public','record','return','sealed','short','static','strictfp','super',
        'switch','synchronized','this','throw','throws','transient','try','var',
        'void','volatile','while','true','false',
    ])
    operands = [i for i in identifiers if i not in _JAVA_KEYWORDS]

    n1 = len(set(operators))   # unique operators
    n2 = len(set(operands))    # unique operands
    N1 = len(operators)        # total operators
    N2 = len(operands)         # total operands

    n  = n1 + n2
    N  = N1 + N2

    if n == 0 or n2 == 0:
        return {"volume": 0.0, "difficulty": 0.0, "effort": 0.0, "bugs": 0.0}

    volume     = N * math.log2(n) if n > 1 else 0.0
    difficulty = (n1 / 2.0) * (N2 / max(n2, 1))
    effort     = difficulty * volume
    bugs       = volume / 3000.0

    return {
        "volume":     round(volume, 3),
        "difficulty": round(difficulty, 3),
        "effort":     round(effort, 3),
        "bugs":       round(bugs, 6),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Maintainability Index (same formula as Python radon)
# ─────────────────────────────────────────────────────────────────────────────

def _maintainability_index(volume: float, cc: float, loc: int) -> float:
    """MI = max(0, (171 - 5.2*ln(V) - 0.23*CC - 16.2*ln(LOC)) * 100/171)"""
    try:
        v_term   = 5.2  * math.log(max(volume, 1.0))
        cc_term  = 0.23 * max(cc, 1.0)
        loc_term = 16.2 * math.log(max(loc, 1))
        mi = (171 - v_term - cc_term - loc_term) * (100.0 / 171.0)
        return round(max(0.0, mi), 2)
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Cognitive Complexity (Sonar-style approximation)
# ─────────────────────────────────────────────────────────────────────────────

def _cognitive_complexity(source: str) -> int:
    """
    Approximate Sonar-style cognitive complexity:
    +1 for each: if, else, else if, for, while, do, switch, case, catch, break (labelled), continue
    +extra for each nesting level above 1
    """
    clean = _strip_comments(source)
    clean = _RE_STRING_LITERAL.sub('""', clean)

    score = 0
    depth = 0
    lines = clean.splitlines()

    _NESTING_INCREASE = re.compile(r'\b(if|else\s+if|else|for|while|do|switch|try|catch|finally|synchronized)\b')
    _FLAT_ADDITIONS   = re.compile(r'\b(break|continue|return|throw)\b')

    for line in lines:
        stripped = line.strip()
        nesting_hits = len(_NESTING_INCREASE.findall(stripped))
        if nesting_hits:
            score += nesting_hits + depth
            depth += nesting_hits
        score += len(_FLAT_ADDITIONS.findall(stripped))
        # Update depth based on brace balance
        depth += stripped.count('{') - stripped.count('}')
        depth = max(0, depth)

    return score


# ─────────────────────────────────────────────────────────────────────────────
# Method Size Extraction (regex-based brace matching)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_method_bodies(source: str) -> list[tuple[int, str, list[str]]]:
    """
    Extract (start_line, method_name, param_names) for each method.
    Returns list of (loc_count, method_name, params_list) using brace matching.
    """
    clean = _strip_comments(source)
    methods = []

    for m in _RE_METHOD_SIG.finditer(clean):
        method_name = m.group(1)
        params_raw  = m.group(2).strip()
        params      = [p.strip() for p in params_raw.split(',') if p.strip()] if params_raw else []

        # Find matching close brace
        start = m.end() - 1  # position of opening {
        depth = 0
        body_start = start
        body_end   = start

        for i in range(start, len(clean)):
            if clean[i] == '{':
                depth += 1
            elif clean[i] == '}':
                depth -= 1
                if depth == 0:
                    body_end = i
                    break

        body = clean[body_start:body_end + 1]
        loc  = len(body.splitlines())
        methods.append((loc, method_name, params))

    return methods


# ─────────────────────────────────────────────────────────────────────────────
# Primary Extractor: javalang AST
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_javalang(source: str) -> Optional[dict]:
    """
    Parse Java source with javalang and extract structured metrics.
    Returns None on parse failure (triggers regex fallback).
    """
    if not HAS_JAVALANG:
        return None

    try:
        tree = javalang.parse.parse(source)
    except Exception:
        return None

    lines = source.splitlines()
    loc   = len(lines)
    clean = _strip_comments(source)
    sloc  = sum(1 for l in clean.splitlines() if l.strip())

    # Classes / interfaces / enums
    class_decls  = list(tree.filter(javalang.tree.ClassDeclaration))
    iface_decls  = list(tree.filter(javalang.tree.InterfaceDeclaration))
    enum_decls   = list(tree.filter(javalang.tree.EnumDeclaration))
    class_count  = len(class_decls) + len(iface_decls) + len(enum_decls)

    # Imports
    import_count = len(tree.imports) if tree.imports else 0

    # Methods
    method_decls = list(tree.filter(javalang.tree.MethodDeclaration))
    function_count = len(method_decls)

    # Use regex method body extraction for sizes/params (javalang doesn't track LOC)
    method_bodies = _extract_method_bodies(source)
    method_locs   = [m[0] for m in method_bodies] if method_bodies else []
    method_params = [len(m[2]) for m in method_bodies] if method_bodies else []

    avg_func_size = round(sum(method_locs) / max(len(method_locs), 1), 2)
    max_func_size = max(method_locs) if method_locs else 0
    avg_param     = round(sum(method_params) / max(len(method_params), 1), 2)
    max_param     = max(method_params) if method_params else 0

    # Complexity
    cc_values = [_cyclomatic_complexity(source[b:e]) for b, e in _get_method_spans(source)]
    if not cc_values:
        cc_values = [1]
    avg_cc = round(sum(cc_values) / len(cc_values), 2)
    max_cc = max(cc_values)

    # Halstead
    h = _halstead_metrics(source)

    # Nesting
    max_nesting = _compute_max_nesting(source)

    # Maintainability
    mi = _maintainability_index(h["volume"], avg_cc, loc)

    # Cognitive complexity
    cog = _cognitive_complexity(source)

    # Comments
    comment_lines   = _count_comment_lines(source)
    blank_lines     = sum(1 for l in lines if not l.strip())
    comment_density = round(comment_lines / max(loc, 1), 4)

    return {
        "loc": loc, "sloc": sloc, "blank_lines": blank_lines,
        "comment_lines": comment_lines, "comment_density": comment_density,
        "function_count": function_count, "class_count": class_count,
        "import_count": import_count,
        "avg_function_size": avg_func_size, "max_function_size": max_func_size,
        "avg_param_count": avg_param, "max_param_count": max_param,
        "max_nesting_depth": max_nesting,
        "avg_cyclomatic_complexity": avg_cc, "max_cyclomatic_complexity": max_cc,
        "halstead_volume": h["volume"], "halstead_difficulty": h["difficulty"],
        "halstead_effort": h["effort"], "halstead_bugs": h["bugs"],
        "maintainability_index": mi, "cognitive_complexity": cog,
        "parse_fallback": False,
    }


def _get_method_spans(source: str) -> list[tuple[int, int]]:
    """Return (start, end) character spans for each method body."""
    clean = _strip_comments(source)
    spans = []
    for m in _RE_METHOD_SIG.finditer(clean):
        start = m.end() - 1
        depth = 0
        for i in range(start, len(clean)):
            if clean[i] == '{':
                depth += 1
            elif clean[i] == '}':
                depth -= 1
                if depth == 0:
                    spans.append((start, i + 1))
                    break
    return spans


# ─────────────────────────────────────────────────────────────────────────────
# Fallback Extractor: pure regex (Java 16+, syntax errors)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_via_regex(source: str) -> dict:
    """Regex-based fallback feature extraction when javalang fails."""
    lines = source.splitlines()
    loc   = len(lines)
    clean = _strip_comments(source)
    sloc  = sum(1 for l in clean.splitlines() if l.strip())

    blank_lines     = sum(1 for l in lines if not l.strip())
    comment_lines   = _count_comment_lines(source)
    comment_density = round(comment_lines / max(loc, 1), 4)

    class_count  = len(_RE_CLASS_DECL.findall(clean))
    import_count = len(_RE_IMPORT.findall(source))

    method_bodies = _extract_method_bodies(source)
    function_count = len(method_bodies)
    method_locs    = [m[0] for m in method_bodies] if method_bodies else []
    method_params  = [len(m[2]) for m in method_bodies] if method_bodies else []

    avg_func_size = round(sum(method_locs) / max(len(method_locs), 1), 2)
    max_func_size = max(method_locs) if method_locs else 0
    avg_param     = round(sum(method_params) / max(len(method_params), 1), 2)
    max_param     = max(method_params) if method_params else 0

    # Approximate CC over whole file
    whole_cc = _cyclomatic_complexity(clean)
    avg_cc   = round(whole_cc / max(function_count, 1), 2)
    max_cc   = whole_cc

    h   = _halstead_metrics(source)
    mi  = _maintainability_index(h["volume"], avg_cc, loc)
    cog = _cognitive_complexity(source)
    max_nesting = _compute_max_nesting(source)

    return {
        "loc": loc, "sloc": sloc, "blank_lines": blank_lines,
        "comment_lines": comment_lines, "comment_density": comment_density,
        "function_count": function_count, "class_count": class_count,
        "import_count": import_count,
        "avg_function_size": avg_func_size, "max_function_size": max_func_size,
        "avg_param_count": avg_param, "max_param_count": max_param,
        "max_nesting_depth": max_nesting,
        "avg_cyclomatic_complexity": avg_cc, "max_cyclomatic_complexity": max_cc,
        "halstead_volume": h["volume"], "halstead_difficulty": h["difficulty"],
        "halstead_effort": h["effort"], "halstead_bugs": h["bugs"],
        "maintainability_index": mi, "cognitive_complexity": cog,
        "parse_fallback": True,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def extract_java_metrics(source: str) -> dict:
    """
    Extract code metrics from Java source code.
    Tries javalang AST first; falls back to regex on parse failure.

    Returns dict with keys matching CODE_METRIC_COLS plus 'parse_fallback'.
    """
    if not source or not source.strip():
        return {
            "loc": 0, "sloc": 0, "blank_lines": 0,
            "comment_lines": 0, "comment_density": 0.0,
            "function_count": 0, "class_count": 0, "import_count": 0,
            "avg_function_size": 0.0, "max_function_size": 0,
            "avg_param_count": 0.0, "max_param_count": 0,
            "max_nesting_depth": 0,
            "avg_cyclomatic_complexity": 0.0, "max_cyclomatic_complexity": 0,
            "halstead_volume": 0.0, "halstead_difficulty": 0.0,
            "halstead_effort": 0.0, "halstead_bugs": 0.0,
            "maintainability_index": 0.0, "cognitive_complexity": 0,
            "parse_fallback": False,
        }

    # Try javalang first
    result = _extract_via_javalang(source)
    if result is not None:
        return result

    # Regex fallback
    logger.debug("javalang parse failed — using regex fallback")
    return _extract_via_regex(source)


def extract_java_smells(source: str, metrics: dict) -> dict:
    """
    Detect code smells from Java source given pre-computed metrics.

    Returns dict with keys matching SMELL_COLS.
    """
    method_bodies = _extract_method_bodies(source)
    method_locs   = [m[0] for m in method_bodies]
    method_params = [len(m[2]) for m in method_bodies]

    long_method_count = sum(1 for loc in method_locs if loc > THRESHOLD_LONG_METHOD)
    if long_method_count == 0 and metrics.get("max_function_size", 0) > THRESHOLD_LONG_METHOD:
        long_method_count = 1

    long_param_count = sum(1 for p in method_params if p > THRESHOLD_LONG_PARAM)
    if long_param_count == 0 and metrics.get("max_param_count", 0) > THRESHOLD_LONG_PARAM:
        long_param_count = 1

    # Class-level large class: approximate per-class LOC from total
    loc_per_class = (metrics.get("loc", 0) / max(metrics.get("class_count", 1), 1))
    large_class_count  = 1 if loc_per_class > THRESHOLD_LARGE_CLASS else 0

    # Deep nesting count: methods with nesting > threshold
    deep_nesting_count = 1 if metrics.get("max_nesting_depth", 0) > THRESHOLD_DEEP_NESTING else 0

    # High complexity count: methods with CC > threshold
    cc_values = [_cyclomatic_complexity(source[b:e]) for b, e in _get_method_spans(source)]
    high_complexity_count = sum(1 for cc in cc_values if cc > THRESHOLD_HIGH_COMPLEXITY)
    if high_complexity_count == 0 and metrics.get("max_cyclomatic_complexity", 0) > THRESHOLD_HIGH_COMPLEXITY:
        high_complexity_count = 1

    has_long_method      = 1 if long_method_count > 0 else 0
    has_long_param_list  = 1 if long_param_count > 0 else 0
    has_large_class      = 1 if large_class_count > 0 else 0
    has_deep_nesting     = 1 if deep_nesting_count > 0 else 0
    has_high_complexity  = 1 if high_complexity_count > 0 else 0

    total_smells = (has_long_method + has_long_param_list + has_large_class +
                    has_deep_nesting + has_high_complexity)

    return {
        "has_long_method":       has_long_method,
        "has_long_param_list":   has_long_param_list,
        "has_large_class":       has_large_class,
        "has_deep_nesting":      has_deep_nesting,
        "has_high_complexity":   has_high_complexity,
        "long_method_count":     long_method_count,
        "long_param_count":      long_param_count,
        "large_class_count":     large_class_count,
        "deep_nesting_count":    deep_nesting_count,
        "high_complexity_count": high_complexity_count,
        "total_smells":          total_smells,
    }
