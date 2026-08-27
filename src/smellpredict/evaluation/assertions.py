"""
SmellPredict — Data Integrity Assertions
==========================================
Hard assertions that MUST pass before any experiment runs.
These enforce the two critical properties corrected in the paper:

  1. CHRONOLOGICAL: max(train_date) < min(test_date)   for every fold
  2. IDENTITY:      train_files ∩ test_files = ∅        for every fold

If any assertion fails, an exception is raised immediately —
results computed on a leaky split are silently wrong and unacceptable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from loguru import logger


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Split Assertions
# ─────────────────────────────────────────────────────────────────────────────

def assert_no_temporal_leak(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_id: int | str,
    date_col: str = "commit_date",   # Fix 0-D: was snapshot_date (string ISO); commit_date is proper datetime
) -> None:
    """
    Assert that ALL training dates precede ALL test dates.
    Raises AssertionError if violated.

    Args:
        train_df: Training fold DataFrame
        test_df: Test fold DataFrame
        fold_id: Fold identifier for error messages
        date_col: Name of the date column
    """
    if train_df.empty or test_df.empty:
        logger.warning(f"Fold {fold_id}: Empty train or test split — skipping date check")
        return

    train_dates = pd.to_datetime(train_df[date_col], utc=True)
    test_dates = pd.to_datetime(test_df[date_col], utc=True)

    max_train = train_dates.max()
    min_test = test_dates.min()

    if max_train >= min_test:
        raise AssertionError(
            f"🚨 TEMPORAL LEAK in fold {fold_id}!\n"
            f"  max(train_date) = {max_train}\n"
            f"  min(test_date)  = {min_test}\n"
            f"  Training snapshots from the FUTURE are present in training set.\n"
            f"  Fix the split logic before proceeding."
        )

    logger.debug(
        f"✅ Fold {fold_id}: Chronological order verified "
        f"(max_train={max_train.date()}, min_test={min_test.date()})"
    )


def assert_no_file_identity_leak(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_id: int | str,
    file_col: str = "canonical_file_id",
) -> None:
    """
    Assert that no file appears in both training and test sets.
    Raises AssertionError if violated.

    Args:
        train_df: Training fold DataFrame
        test_df: Test fold DataFrame
        fold_id: Fold identifier
        file_col: Column containing file identity key
    """
    train_files = set(train_df[file_col].unique())
    test_files = set(test_df[file_col].unique())
    overlap = train_files & test_files

    if overlap:
        raise AssertionError(
            f"🚨 FILE IDENTITY LEAK in fold {fold_id}!\n"
            f"  {len(overlap)} files appear in BOTH train and test.\n"
            f"  Sample overlapping files: {list(overlap)[:5]}\n"
            f"  Fix the split logic before proceeding."
        )

    logger.debug(
        f"✅ Fold {fold_id}: File identity separation verified "
        f"(train_files={len(train_files)}, test_files={len(test_files)}, overlap=0)"
    )


def assert_no_preprocessing_leak(
    transformer,
    was_fit_on_train_only: bool,
    fold_id: int | str,
) -> None:
    """
    Soft assertion to verify preprocessing was fit on training data only.
    Call this after fitting any scaler/imputer to document compliance.
    """
    if not was_fit_on_train_only:
        raise AssertionError(
            f"🚨 PREPROCESSING LEAK in fold {fold_id}!\n"
            f"  Transformer {type(transformer).__name__} was NOT fit on training data only.\n"
            f"  This leaks test statistics into training. Fix immediately."
        )
    logger.debug(f"✅ Fold {fold_id}: Preprocessing fit on train-only confirmed")


def assert_lopo_integrity(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    held_out_repo: str,
    repo_col: str = "repo",
) -> None:
    """
    Assert that the held-out repository never appears in the LOPO training set.

    Args:
        train_df: Training DataFrame (should NOT contain held_out_repo)
        test_df: Test DataFrame (should ONLY contain held_out_repo)
        held_out_repo: The repository name being held out
        repo_col: Column containing repository name
    """
    train_repos = set(train_df[repo_col].unique())
    test_repos = set(test_df[repo_col].unique())

    if held_out_repo in train_repos:
        raise AssertionError(
            f"🚨 LOPO INTEGRITY VIOLATION!\n"
            f"  Held-out repo '{held_out_repo}' appears in TRAINING set.\n"
            f"  Train repos: {sorted(train_repos)}\n"
            f"  Fix the LOPO split logic immediately."
        )

    if test_repos != {held_out_repo}:
        raise AssertionError(
            f"🚨 LOPO TEST SET CONTAMINATION!\n"
            f"  Test set should only contain '{held_out_repo}'.\n"
            f"  Actual test repos: {sorted(test_repos)}"
        )

    logger.debug(
        f"✅ LOPO integrity: '{held_out_repo}' correctly excluded from training. "
        f"Training repos: {sorted(train_repos)}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Full Fold Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_id: int | str,
    date_col: str = "commit_date",   # Fix 0-D: was snapshot_date
    file_col: str = "canonical_file_id",
    min_train_rows: int = 50,
    min_test_rows: int = 10,
) -> dict:
    """
    Run all integrity checks for a single fold. Raises on any violation.

    Args:
        train_df: Training fold
        test_df: Test fold
        fold_id: Identifier for logging
        date_col: Date column name
        file_col: File identity column name
        min_train_rows: Minimum rows required in training set
        min_test_rows: Minimum rows required in test set

    Returns:
        Dict with fold statistics
    """
    # Size checks
    if len(train_df) < min_train_rows:
        raise AssertionError(
            f"Fold {fold_id}: Training set too small ({len(train_df)} < {min_train_rows})"
        )
    if len(test_df) < min_test_rows:
        raise AssertionError(
            f"Fold {fold_id}: Test set too small ({len(test_df)} < {min_test_rows})"
        )

    # Temporal integrity
    assert_no_temporal_leak(train_df, test_df, fold_id, date_col)

    # File identity integrity
    assert_no_file_identity_leak(train_df, test_df, fold_id, file_col)

    # Report fold stats
    train_pos = train_df["future_bug_fix"].sum() if "future_bug_fix" in train_df.columns else "N/A"
    test_pos = test_df["future_bug_fix"].sum() if "future_bug_fix" in test_df.columns else "N/A"

    stats = {
        "fold_id": fold_id,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "train_files": train_df[file_col].nunique(),
        "test_files": test_df[file_col].nunique(),
        "train_positives": int(train_pos) if isinstance(train_pos, (int, float)) else train_pos,
        "test_positives": int(test_pos) if isinstance(test_pos, (int, float)) else test_pos,
        "max_train_date": str(pd.to_datetime(train_df[date_col], utc=True).max().date()),
        "min_test_date": str(pd.to_datetime(test_df[date_col], utc=True).min().date()),
        "no_overlap": True,
        "chronologically_valid": True,
    }

    logger.info(
        f"  Fold {fold_id} validated: "
        f"train={stats['train_rows']}r/{stats['train_files']}f, "
        f"test={stats['test_rows']}r/{stats['test_files']}f"
    )
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Temporal Split Generator
# ─────────────────────────────────────────────────────────────────────────────

def temporal_split_generator(
    df: pd.DataFrame,
    n_folds: int = 10,
    date_col: str = "commit_date",   # Fix 0-D: was snapshot_date
    file_col: str = "canonical_file_id",
    label_col: str = "future_bug_fix",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, dict]]:
    """
    Generate chronologically-valid, file-identity-exclusive temporal folds.

    Uses an expanding-window strategy:
      - Divide the date range into n_folds+1 intervals
      - Fold k: train on everything before cutoff[k], test on cutoff[k] to cutoff[k+1]
      - Remove from test any file that appeared in training

    Args:
        df: Full dataset
        n_folds: Number of temporal folds
        date_col: Date column
        file_col: File identity column
        label_col: Label column (for reporting positive rates)

    Yields:
        (train_df, test_df, stats_dict) for each valid fold
    """
    from typing import Iterator

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values(date_col)

    min_date = df[date_col].min()
    max_date = df[date_col].max()

    # Compute n_folds+1 boundary dates
    total_seconds = (max_date - min_date).total_seconds()
    boundaries = [
        min_date + pd.Timedelta(seconds=total_seconds * i / n_folds)
        for i in range(n_folds + 1)
    ]

    fold_id = 0
    for i in range(n_folds - 1):
        train_cutoff = boundaries[i + 1]
        test_start = boundaries[i + 1]
        test_end = boundaries[i + 2]

        train_df = df[df[date_col] <= train_cutoff].copy()
        test_df = df[(df[date_col] > test_start) & (df[date_col] <= test_end)].copy()

        if train_df.empty or test_df.empty:
            continue

        # Identity exclusion: remove test files seen in training
        train_file_ids = set(train_df[file_col].unique())
        test_df = test_df[~test_df[file_col].isin(train_file_ids)].copy()

        if test_df.empty:
            logger.warning(f"Fold {fold_id}: All test files appeared in training — skipping")
            continue

        try:
            stats = validate_fold(train_df, test_df, fold_id, date_col, file_col)
            fold_id += 1
            yield train_df, test_df, stats
        except AssertionError as e:
            logger.error(f"Fold validation FAILED: {e}")
            raise


def lopo_split_generator(
    df: pd.DataFrame,
    repo_col: str = "repo",
    date_col: str = "commit_date",   # Fix 0-D: was snapshot_date
    file_col: str = "canonical_file_id",
) -> Iterator[tuple[pd.DataFrame, pd.DataFrame, str]]:
    """
    Generate Leave-One-Project-Out (LOPO) splits.
    Each rotation holds out one repository as the test set.

    Yields:
        (train_df, test_df, held_out_repo_name)
    """
    from typing import Iterator

    repos = sorted(df[repo_col].unique())
    logger.info(f"LOPO: {len(repos)} rotations over repos: {repos}")

    for held_out in repos:
        train_df = df[df[repo_col] != held_out].copy()
        test_df = df[df[repo_col] == held_out].copy()

        if train_df.empty or test_df.empty:
            logger.warning(f"LOPO: Skipping {held_out} — empty split")
            continue

        assert_lopo_integrity(train_df, test_df, held_out, repo_col)
        logger.info(
            f"LOPO: Held-out={held_out}, "
            f"train={len(train_df)} rows from {len(repos)-1} repos, "
            f"test={len(test_df)} rows"
        )
        yield train_df, test_df, held_out
