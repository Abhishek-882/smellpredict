"""
SmellPredict — Bug-Fix Labeling Module
========================================
Implements the commit-message heuristic from the paper (Equation 1):

  score = 0.20·min(k,3) + 0.15·min(i,2) + 0.25·min(s,3) − 0.15·min(n,2)

  k = count of fix-related keywords
  i = count of issue-reference patterns
  s = count of strong fix-intent regex matches
  n = count of noise patterns

A commit is labeled bug-fix when:
  score ≥ threshold  AND  NOT (n ≥ 2 AND s == 0)

This module also:
  - Generates stratified annotation samples for human validation
  - Calculates Cohen's κ from annotation results
  - Supports LLM-assisted labeling as an ensemble signal
"""

from __future__ import annotations

import re
import math
import random
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Constants (from paper)
# ─────────────────────────────────────────────────────────────────────────────

KEYWORD_WEIGHT = 0.20
ISSUE_REF_WEIGHT = 0.15
STRONG_FIX_WEIGHT = 0.25
NOISE_PENALTY = 0.15
DEFAULT_THRESHOLD = 0.40

FIX_KEYWORDS = frozenset([
    "fix", "fixed", "fixes", "fixing", "bug", "defect", "patch",
    "regression", "crash", "exception", "failure", "broken", "incorrect",
    "wrong", "error", "resolve", "resolved", "resolves", "resolving",
    "repair", "hotfix",
])

STRONG_PATTERNS = [
    re.compile(r"\bfix(es|ed|ing)?\b", re.IGNORECASE),
    re.compile(r"\bbug.?fix\b", re.IGNORECASE),
    re.compile(r"\bregression\b", re.IGNORECASE),
    # NOTE: 'patch' removed from strong patterns — Apache SVN uses "patch by X;
    # reviewed by Y" for ALL commit types (features, refactors, moves), so
    # treating it as a strong signal causes massive false-positives on
    # Cassandra-style repos. It remains a low-weight keyword in FIX_KEYWORDS.
    re.compile(r"\bcrash\b", re.IGNORECASE),
    re.compile(r"\bdefect\b", re.IGNORECASE),
    re.compile(r"\bnull.?pointer\b", re.IGNORECASE),
    re.compile(r"\bnpe\b", re.IGNORECASE),
    re.compile(r"\bstack.?overflow\b", re.IGNORECASE),
    re.compile(r"\bdeadlock\b", re.IGNORECASE),
    re.compile(r"\bmemory.?leak\b", re.IGNORECASE),
]

ISSUE_REF_PATTERNS = [
    re.compile(r"#\d+"),
    re.compile(r"closes\s+#\d+", re.IGNORECASE),
    re.compile(r"fixes\s+#\d+", re.IGNORECASE),
    re.compile(r"resolves\s+#\d+", re.IGNORECASE),
    re.compile(r"gh-\d+", re.IGNORECASE),
    # Fix 0-C: Jira-style issue references used by Apache, Elastic, and many OSS projects.
    # Examples: ELASTICSEARCH-1234, CASSANDRA-5678, KAFKA-999, SPARK-12345
    # Without this, repos using Jira (not GitHub Issues) score 0% bug labels.
    re.compile(r"\b[A-Z]{2,12}-\d{1,6}\b"),
]

NOISE_PATTERNS = [
    re.compile(r"\bmerge\b", re.IGNORECASE),
    re.compile(r"\bformat\b", re.IGNORECASE),
    re.compile(r"\btypo\b", re.IGNORECASE),
    re.compile(r"\breadme\b", re.IGNORECASE),
    re.compile(r"\bdocumentation\b", re.IGNORECASE),
    re.compile(r"\bdocs\b", re.IGNORECASE),
    re.compile(r"\brefactor\b", re.IGNORECASE),
    re.compile(r"\bcleanup\b", re.IGNORECASE),
    re.compile(r"\bstyle\b", re.IGNORECASE),
    re.compile(r"\blint\b", re.IGNORECASE),
    # Apache SVN workflow noise — these appear in ALL Apache commits regardless
    # of whether the change is a bug fix, feature, or structural rename.
    re.compile(r"patch\s+by\b", re.IGNORECASE),
    re.compile(r"reviewed\s+by\b", re.IGNORECASE),
    re.compile(r"git-svn-id\b", re.IGNORECASE),
    # Structural / non-bug commits
    re.compile(r"\bmove\b", re.IGNORECASE),
    re.compile(r"\brename\b", re.IGNORECASE),
    re.compile(r"\brevert\b", re.IGNORECASE),
    re.compile(r"\bupgrade\b", re.IGNORECASE),
    re.compile(r"\bupdate\s+version\b", re.IGNORECASE),
    re.compile(r"\bbump\b", re.IGNORECASE),
]


# ─────────────────────────────────────────────────────────────────────────────
# Core Labeling
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CommitScore:
    message: str
    score: float
    k: int                    # keyword count (capped at 3)
    i: int                    # issue ref count (capped at 2)
    s: int                    # strong pattern count (capped at 3)
    n: int                    # noise count (capped at 2)
    is_bug_fix: bool
    noise_dominated: bool     # n≥2 AND s==0 → force negative


def score_commit(
    message: str,
    threshold: float = DEFAULT_THRESHOLD,
) -> CommitScore:
    """
    Score a commit message using the paper's heuristic (Equation 1).

    Args:
        message: Raw commit message string
        threshold: Score threshold above which commit is labeled bug-fix

    Returns:
        CommitScore with all intermediate values and final label
    """
    msg = message.lower().strip()
    tokens = set(re.findall(r"\b\w+\b", msg))

    # k: fix-related keywords
    k = min(len(tokens & FIX_KEYWORDS), 3)

    # i: issue references
    i_raw = sum(1 for p in ISSUE_REF_PATTERNS if p.search(msg))
    i = min(i_raw, 2)

    # s: strong fix-intent patterns
    s_raw = sum(1 for p in STRONG_PATTERNS if p.search(msg))
    s = min(s_raw, 3)

    # n: noise patterns
    n_raw = sum(1 for p in NOISE_PATTERNS if p.search(msg))
    n = min(n_raw, 2)

    score = (
        KEYWORD_WEIGHT * k
        + ISSUE_REF_WEIGHT * i
        + STRONG_FIX_WEIGHT * s
        - NOISE_PENALTY * n
    )
    score = max(0.0, min(1.0, score))  # clip to [0, 1]

    noise_dominated = (n >= 2 and s == 0)
    is_bug_fix = (score >= threshold) and (not noise_dominated)

    return CommitScore(
        message=message,
        score=round(score, 4),
        k=k, i=i, s=s, n=n,
        is_bug_fix=is_bug_fix,
        noise_dominated=noise_dominated,
    )


def label_commits(
    messages: list[str],
    threshold: float = DEFAULT_THRESHOLD,
) -> list[CommitScore]:
    """Label a list of commit messages."""
    return [score_commit(m, threshold) for m in messages]


# ─────────────────────────────────────────────────────────────────────────────
# Stratified Sampling for Human Annotation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AnnotationSample:
    commit_hash: str
    repository: str
    message: str
    score: float
    stratum: str              # "high_conf_positive", "boundary", "high_conf_negative", "near_miss"
    heuristic_label: bool


def build_annotation_sample(
    commits: list[dict],              # list of {hash, repo, message}
    n_high_pos: int = 100,
    n_boundary: int = 150,
    n_high_neg: int = 150,
    n_near_miss: int = 100,
    seed: int = 42,
) -> list[AnnotationSample]:
    """
    Build a stratified sample of commits for human annotation.

    Strata:
        high_conf_positive  → score ≥ 0.70  (likely true positives)
        boundary            → 0.40 ≤ score < 0.70
        high_conf_negative  → score < 0.40 (likely true negatives)
        near_miss           → score < 0.40 but keyword count > 0

    Args:
        commits: Raw commit dicts with hash, repo, message
        n_*: Target sample size per stratum

    Returns:
        Stratified annotation sample list
    """
    random.seed(seed)
    scored = [(c, score_commit(c["message"])) for c in commits]

    strata: dict[str, list] = {
        "high_conf_positive": [],
        "boundary": [],
        "high_conf_negative": [],
        "near_miss": [],
    }

    for commit, cs in scored:
        if cs.score >= 0.70:
            strata["high_conf_positive"].append((commit, cs))
        elif cs.score >= 0.40:
            strata["boundary"].append((commit, cs))
        elif cs.k > 0:
            strata["near_miss"].append((commit, cs))
        else:
            strata["high_conf_negative"].append((commit, cs))

    targets = {
        "high_conf_positive": n_high_pos,
        "boundary": n_boundary,
        "high_conf_negative": n_high_neg,
        "near_miss": n_near_miss,
    }

    samples: list[AnnotationSample] = []
    for stratum, target_n in targets.items():
        pool = strata[stratum]
        chosen = random.sample(pool, min(target_n, len(pool)))
        for commit, cs in chosen:
            samples.append(AnnotationSample(
                commit_hash=commit["hash"],
                repository=commit["repo"],
                message=commit["message"],
                score=cs.score,
                stratum=stratum,
                heuristic_label=cs.is_bug_fix,
            ))

    logger.info(f"Annotation sample: {len(samples)} commits across {len(strata)} strata")
    for s, pool in strata.items():
        logger.info(f"  {s}: {min(targets[s], len(pool))} / {len(pool)} available")

    return samples


def export_annotation_sample(
    samples: list[AnnotationSample],
    output_path: str | Path,
) -> None:
    """Export annotation samples to JSON for the labeling interface."""
    data = [
        {
            "id": i,
            "commit_hash": s.commit_hash,
            "repository": s.repository,
            "message": s.message,
            "score": s.score,
            "stratum": s.stratum,
            "heuristic_label": s.heuristic_label,
            "human_label": None,          # to be filled by annotators
            "annotator_confidence": None,
            "annotator_notes": None,
        }
        for i, s in enumerate(samples)
    ]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Annotation sample written to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Inter-Rater Agreement
# ─────────────────────────────────────────────────────────────────────────────

def cohens_kappa(
    rater_a: list[bool],
    rater_b: list[bool],
) -> dict:
    """
    Calculate Cohen's κ for two binary raters.

    Returns dict with:
        kappa: Cohen's κ coefficient
        po: observed agreement
        pe: expected agreement by chance
        agreement_matrix: 2x2 confusion-style matrix
        interpretation: qualitative label
    """
    assert len(rater_a) == len(rater_b), "Raters must have equal number of ratings"
    n = len(rater_a)

    tp = sum(1 for a, b in zip(rater_a, rater_b) if a and b)
    tn = sum(1 for a, b in zip(rater_a, rater_b) if not a and not b)
    fp = sum(1 for a, b in zip(rater_a, rater_b) if not a and b)
    fn = sum(1 for a, b in zip(rater_a, rater_b) if a and not b)

    po = (tp + tn) / n

    # Marginal proportions
    p_pos_a = (tp + fn) / n
    p_neg_a = (fp + tn) / n
    p_pos_b = (tp + fp) / n
    p_neg_b = (fn + tn) / n

    pe = (p_pos_a * p_pos_b) + (p_neg_a * p_neg_b)

    if pe == 1.0:
        kappa = 1.0
    else:
        kappa = (po - pe) / (1 - pe)

    # Interpretation (Landis & Koch scale)
    if kappa < 0:
        interp = "Poor (below chance)"
    elif kappa < 0.20:
        interp = "Slight"
    elif kappa < 0.40:
        interp = "Fair"
    elif kappa < 0.60:
        interp = "Moderate"
    elif kappa < 0.80:
        interp = "Substantial"
    else:
        interp = "Almost perfect"

    return {
        "kappa": round(kappa, 4),
        "po": round(po, 4),
        "pe": round(pe, 4),
        "agreement_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "n": n,
        "interpretation": interp,
    }


def heuristic_vs_human_metrics(
    heuristic_labels: list[bool],
    human_labels: list[bool],
) -> dict:
    """
    Calculate precision, recall, F1 of heuristic against human ground truth.

    Args:
        heuristic_labels: Model-generated labels (predictions)
        human_labels: Human-annotated labels (ground truth)

    Returns:
        Dict with precision, recall, f1, kappa, support
    """
    tp = sum(1 for h, g in zip(heuristic_labels, human_labels) if h and g)
    fp = sum(1 for h, g in zip(heuristic_labels, human_labels) if h and not g)
    fn = sum(1 for h, g in zip(heuristic_labels, human_labels) if not h and g)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    kappa_result = cohens_kappa(heuristic_labels, human_labels)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "n_positive_human": sum(human_labels),
        "n_positive_heuristic": sum(heuristic_labels),
        "kappa": kappa_result["kappa"],
        "kappa_interpretation": kappa_result["interpretation"],
        "support": len(heuristic_labels),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_messages = [
        "Fix null pointer exception in core.py (#1234)",
        "Add new feature: user authentication",
        "Fix crash when input is empty",
        "Merge pull request #42 from dev/feature-branch",
        "Refactor database connection logic",
        "bug fix: resolved issue with timeout regression",
        "Update documentation for API endpoints",
        "hotfix: patch for broken login endpoint",
    ]

    print("=" * 60)
    print("SmellPredict — Bug-Fix Labeling Demo")
    print("=" * 60)
    for msg in test_messages:
        cs = score_commit(msg)
        label = "✅ BUG-FIX" if cs.is_bug_fix else "❌ NOT BUG-FIX"
        print(f"{label}  (score={cs.score:.3f})  |  {msg[:60]}")

    # Kappa demo
    rater_a = [True, True, False, True, False, True, False, False]
    rater_b = [True, True, False, False, False, True, True, False]
    result = cohens_kappa(rater_a, rater_b)
    print(f"\nCohen's κ demo: κ={result['kappa']}, interpretation={result['interpretation']}")
