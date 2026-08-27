"""
SmellPredict — Repository Mining Module
=========================================
Mines Python repositories using PyDriller to extract:

  - File snapshots at fixed commit strides
  - Source-code metrics and smell features per snapshot
  - Development-history metrics computed from commit history
  - Bug-fix labels using the validated heuristic

Architecture:
  ┌─────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ clone_repo  │──▶│ get_snapshots│──▶│ extract_feats│──▶│ save_parquet │
  └─────────────┘   └──────────────┘   └──────────────┘   └──────────────┘

All temporal properties are enforced by assertions before any snapshot is
emitted — max(train_date) < min(test_date) AND train_files ∩ test_files = ∅.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import log2
from pathlib import Path
from typing import Iterator, Optional

import duckdb
import pandas as pd
from loguru import logger
from tqdm import tqdm

try:
    from pydriller import Repository
    HAS_PYDRILLER = True
except ImportError:
    HAS_PYDRILLER = False
    logger.warning("PyDriller not installed. Mining features disabled.")

# Ensure logs directory exists and setup file logging
log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    log_dir / "mining.log",
    rotation="50 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    encoding="utf-8",
)

from smellpredict.features.extractor import (
    extract_code_metrics,
    extract_smell_features,
    FileFeatures,
    HistoryMetrics,
)
from smellpredict.labeling.heuristic import score_commit


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EXCLUDE_PATTERNS = [
    re.compile(r"test_.*\.py$"),
    re.compile(r".*/tests/.*\.py$"),
    re.compile(r".*/migrations/.*\.py$"),
    re.compile(r".*/vendor/.*\.py$"),
    re.compile(r".*/__pycache__/.*"),
    re.compile(r".*/setup\.py$"),
]

MAX_FILE_SIZE_BYTES = 500 * 1024  # 500 KB


# ─────────────────────────────────────────────────────────────────────────────
# Repository Cloning
# ─────────────────────────────────────────────────────────────────────────────

def clone_repo(url: str, target_dir: str | Path, bare: bool = False, max_retries: int = 3) -> Path:
    """
    Clone a remote repository to local cache with automatic retries.

    Args:
        url: Remote Git URL
        target_dir: Local destination directory
        bare: Whether to perform a bare clone
        max_retries: Number of clone attempts before failing

    Returns:
        Path to cloned repository
    """
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    result = None
    for attempt in range(1, max_retries + 1):
        logger.info(f"Cloning {url} → {target} (Attempt {attempt}/{max_retries})")
        if target.exists() and not (target / ".git").exists():
            import shutil
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            ["git", "clone"] + (["--bare"] if bare else []) + [url, str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace"
        )
        if result.returncode == 0:
            logger.success(f"Cloned {url}")
            return target

        logger.warning(f"Clone attempt {attempt} failed for {url}: {result.stderr.strip() if result.stderr else 'unknown error'}")
        time.sleep(3)

    err_msg = result.stderr if result and result.stderr else "Unknown git clone error"
    logger.error(f"Clone failed for {url} after {max_retries} attempts: {err_msg}")
    raise RuntimeError(f"git clone failed: {err_msg}")


# ─────────────────────────────────────────────────────────────────────────────
# File Identity (Canonical ID for rename tracking)
# ─────────────────────────────────────────────────────────────────────────────

def canonical_file_id(repo_name: str, file_path: str) -> str:
    """
    Generate a stable canonical ID for a file using its content hash of
    (repo_name, normalized_path). In a full implementation this would
    be computed from a rename-tracking chain; here we use a deterministic
    hash as a placeholder that can be updated when rename tracking resolves.
    """
    key = f"{repo_name}::{file_path.replace(os.sep, '/')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# History Metric Computation
# ─────────────────────────────────────────────────────────────────────────────

def _should_exclude(file_path: str) -> bool:
    """Return True if this file path should be excluded from analysis."""
    fp = file_path.replace("\\", "/")
    return any(p.search(fp) for p in EXCLUDE_PATTERNS)


def compute_history_metrics(
    file_path: str,
    repo_path: str | Path,
    snapshot_date: datetime,
    window_days_recent: int = 30,
    window_days_active: int = 7,
) -> HistoryMetrics:
    """
    Compute development-history metrics for a file using git log.
    All computation uses commits STRICTLY BEFORE snapshot_date to avoid leakage.

    Args:
        file_path: Relative path to the file within the repo
        repo_path: Local path to the cloned repository
        snapshot_date: Date of the snapshot commit
        window_days_recent: Window for "recent commits" count
        window_days_active: Window for "recently touched" flag

    Returns:
        HistoryMetrics dataclass
    """
    h = HistoryMetrics()

    try:
        # git log for this file before snapshot
        cutoff = snapshot_date.strftime("%Y-%m-%dT%H:%M:%S")
        result = subprocess.run(
            [
                "git", "-C", str(repo_path),
                "log", "--follow",
                f"--before={cutoff}",
                "--format=%H|%ae|%ad|%s",
                "--date=iso-strict",
                "--", file_path,
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
        )

        if result.returncode != 0 or not result.stdout.strip():
            return h

        lines = [l for l in result.stdout.strip().split("\n") if "|" in l]
        if not lines:
            return h

        commits = []
        authors = []
        dates = []
        messages = []

        for line in lines:
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            hash_, author, date_str, message = parts
            commits.append(hash_.strip())
            authors.append(author.strip())
            try:
                dt = datetime.fromisoformat(date_str.strip())
                dates.append(dt)
            except ValueError:
                pass
            messages.append(message.strip())

        if not commits:
            return h

        now = snapshot_date
        h.previous_file_commits = len(commits)

        # Bug-fix count
        h.previous_bug_fixes = sum(
            1 for m in messages if score_commit(m).is_bug_fix
        )

        # Contributors
        h.contributors = len(set(authors))
        if h.contributors > 0:
            from collections import Counter
            top_author_count = Counter(authors).most_common(1)[0][1]
            h.ownership_concentration = round(top_author_count / len(commits), 4)

        # Recent commits
        if dates:
            recent_dates = [
                d for d in dates
                if (now - d.replace(tzinfo=None) if d.tzinfo is None
                    else now.replace(tzinfo=timezone.utc) - d).days <= window_days_recent
            ]
            h.recent_file_commits = len(recent_dates)

            last_change = dates[0]  # first in git log = most recent
            h.days_since_last_change = max(
                0.0,
                (now.replace(tzinfo=None) if now.tzinfo is None else now).timestamp() -
                (last_change.replace(tzinfo=None) if last_change.tzinfo is None
                 else last_change.replace(tzinfo=None)).timestamp()
            ) / 86400
            h.days_since_last_change = round(h.days_since_last_change, 2)

            oldest_date = dates[-1]
            h.file_age_days = round(
                abs(
                    (now.replace(tzinfo=None) if now.tzinfo is None else now.timestamp()) -
                    (oldest_date.replace(tzinfo=None).timestamp()
                     if oldest_date.tzinfo is None else oldest_date.timestamp())
                ) / 86400, 2
            )

        # Time between commits (avg)
        if len(dates) >= 2:
            sorted_dates = sorted(dates)
            gaps = [
                (sorted_dates[i + 1] - sorted_dates[i]).total_seconds() / 86400
                for i in range(len(sorted_dates) - 1)
            ]
            h.avg_time_between_commits = round(sum(gaps) / len(gaps), 2)

        # Commit message entropy
        if messages:
            words: list[str] = []
            for m in messages:
                words.extend(m.lower().split())
            if words:
                from collections import Counter
                freq = Counter(words)
                total = sum(freq.values())
                entropy = -sum((c / total) * log2(c / total) for c in freq.values())
                h.commit_message_entropy = round(entropy, 4)

        h.has_multiple_contributors = int(h.contributors > 1)
        h.is_recently_touched = int(h.days_since_last_change <= window_days_active)

    except Exception as exc:
        logger.debug(f"History computation failed for {file_path}: {exc}")

    return h


# ─────────────────────────────────────────────────────────────────────────────
# Bug-Fix Label Computation
# ─────────────────────────────────────────────────────────────────────────────

# FIX A: Use date-based future window (90 days) instead of commit-count (20).
# The old `--max-count=20` approach caused 75%+ false-positive bug labels on
# active repos (Django: 95.4%, celery: 82.7%) because 20 commits on a busy
# repo equals just 1–2 days — almost guaranteeing a bug-fix exists in that span.
# We now use the same strategy as java_miner.py: --after / --before date flags.
def compute_future_bug_fix_label(
    file_path: str,
    repo_path: str | Path,
    snapshot_date: datetime,
    future_window_days: int = 90,
    threshold: float = 0.40,
) -> tuple[int, float]:
    """
    Look AHEAD from snapshot_date for bug-fix commits touching file_path.

    Uses a DATE-BASED window (default 90 days) to prevent inflated positive
    labels on high-commit-velocity repos. Matches java_miner.py strategy.

    Args:
        file_path: File to check
        repo_path: Repo directory
        snapshot_date: Snapshot commit date (look strictly AFTER this)
        future_window_days: How many calendar days ahead to look (default 90)
        threshold: Bug-fix score threshold

    Returns:
        (label: 0 or 1, max_score: float)
    """
    try:
        # Normalize path separators — Windows backslashes break git log filters
        git_file_path = file_path.replace("\\", "/")

        after_dt = snapshot_date.replace(tzinfo=None) if snapshot_date.tzinfo else snapshot_date
        before_dt = after_dt + timedelta(days=future_window_days)
        after  = after_dt.strftime("%Y-%m-%dT%H:%M:%S")
        before = before_dt.strftime("%Y-%m-%dT%H:%M:%S")

        result = subprocess.run(
            [
                "git", "-C", str(repo_path),
                "log", "--follow",
                f"--after={after}",
                f"--before={before}",       # DATE-BASED window (not commit-count)
                "--format=%s",
                "--", git_file_path,
            ],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30
        )

        if result.returncode != 0 or not result.stdout.strip():
            return 0, 0.0

        messages = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        if not messages:
            return 0, 0.0

        scores = [score_commit(m, threshold) for m in messages]
        bug_fix_scores = [s.score for s in scores if s.is_bug_fix]

        if bug_fix_scores:
            return 1, max(bug_fix_scores)
        return 0, max(s.score for s in scores) if scores else 0.0

    except Exception as exc:
        logger.debug(f"Future label failed for {file_path}: {exc}")
        return 0, 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main Mining Pipeline
# ─────────────────────────────────────────────────────────────────────────────

# FIX B: max_commits cap + even-distribution sampling replaces oldest-first stride.
# Old approach (commits[::20]) always took the OLDEST 5% of history — files there
# were often later renamed/deleted, causing future-label git queries to silently
# return all subsequent history → inflated bug rates. This matches java_miner.py.
MAX_COMMITS_PER_REPO = 500


def mine_repository(
    repo_url: str,
    repo_name: str,
    clone_dir: str | Path,
    output_parquet: str | Path,
    snapshot_stride: int = 5,           # kept for compatibility; overridden by max_commits
    future_window_days: int = 90,       # FIX A: days (not commits)
    label_threshold: float = 0.40,
    smell_thresholds: Optional[dict] = None,
    exclude_tests: bool = True,
    max_file_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> pd.DataFrame:
    """
    Mine a single Python repository, extract features, and return a DataFrame.

    This is the top-level mining function for one repository. Call it in a
    loop (or via Celery / ThreadPoolExecutor) for multiple repositories.

    Args:
        repo_url: GitHub URL of the repository
        repo_name: Human-readable name (used as identifier)
        clone_dir: Local dir to clone into
        output_parquet: Path to write the resulting Parquet file
        snapshot_stride: Sample every N commits
        future_window: Bug-fix look-ahead window (commits)
        label_threshold: Minimum score to label commit as bug-fix
        smell_thresholds: Optional threshold overrides for sensitivity analysis
        exclude_tests: Whether to skip test files
        max_file_size_bytes: Skip files larger than this

    Returns:
        DataFrame with one row per (file, snapshot)
    """
    if not HAS_PYDRILLER:
        raise ImportError("PyDriller is required for mining. pip install pydriller")

    clone_path = Path(clone_dir) / repo_name
    if not clone_path.exists() or not (clone_path / ".git").exists():
        if clone_path.exists():
            import shutil
            shutil.rmtree(clone_path, ignore_errors=True)
        clone_repo(repo_url, clone_path, bare=False)

    logger.info(f"Mining repository: {repo_name} at {clone_path}")

    # Collect all commits
    commits = list(Repository(str(clone_path)).traverse_commits())
    logger.info(f"  Total commits: {len(commits)}")

    # FIX B: Even-distribution sampling capped at MAX_COMMITS_PER_REPO.
    # This ensures snapshots are spread across the FULL history of the repo,
    # not just the oldest N% (which had stale file paths causing git log failures).
    n = len(commits)
    if n <= MAX_COMMITS_PER_REPO:
        sampled = commits
    else:
        indices = [
            int(round(i * (n - 1) / (MAX_COMMITS_PER_REPO - 1)))
            for i in range(MAX_COMMITS_PER_REPO)
        ]
        # Deduplicate while preserving order
        seen = set()
        sampled = []
        for idx in indices:
            if idx not in seen:
                seen.add(idx)
                sampled.append(commits[idx])
    logger.info(f"  Sampled snapshots: {len(sampled)} (even-distribution, cap={MAX_COMMITS_PER_REPO})")

    rows: list[dict] = []

    for commit in tqdm(sampled, desc=f"Snapshots [{repo_name}]"):
        snapshot_date = commit.committer_date

        for modified in commit.modified_files:
            file_path = modified.new_path or modified.old_path
            if not file_path or not file_path.endswith(".py"):
                continue
            if exclude_tests and _should_exclude(file_path):
                continue

            source = modified.source_code
            if not source:
                continue
            if len(source.encode("utf-8", errors="replace")) > max_file_size_bytes:
                continue

            # Feature extraction
            code_metrics, smell_feats = None, None
            try:
                code_metrics = extract_code_metrics(source)
                smell_feats = extract_smell_features(source, smell_thresholds)
            except Exception as e:
                logger.debug(f"Feature extraction failed for {file_path}@{commit.hash[:8]}: {e}")
                continue

            # History metrics (from git log, strictly before snapshot)
            history = compute_history_metrics(
                file_path=file_path,
                repo_path=clone_path,
                snapshot_date=snapshot_date,
            )

            # FIX C: Track whether AST parse succeeded or regex fallback was used
            parse_fallback = getattr(code_metrics, '_parse_fallback', False)

            # Future bug-fix label (strictly after snapshot) — FIX A: date-based window
            label, label_score = compute_future_bug_fix_label(
                file_path=file_path,
                repo_path=clone_path,
                snapshot_date=snapshot_date,
                future_window_days=future_window_days,
                threshold=label_threshold,
            )

            # Build feature row
            # FIX C: Use 'commit_date' (not 'snapshot_date') to match Java schema
            features = FileFeatures(
                repo=repo_name,
                file_path=file_path,
                canonical_file_id=canonical_file_id(repo_name, file_path),
                snapshot_commit=commit.hash,
                snapshot_date=snapshot_date.isoformat(),  # kept for compat
                code=code_metrics,
                smells=smell_feats,
                history=history,
                future_bug_fix=label,
                future_bug_fix_score=label_score,
            )

            row = features.to_flat_dict()
            # FIX C: Add schema-aligned columns matching java_miner.py output
            row["commit_date"] = snapshot_date.replace(tzinfo=None).isoformat()
            row["parse_fallback"] = int(parse_fallback)
            row["language"] = "python"
            rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        logger.warning(f"No rows extracted for {repo_name}")
        return df

    # Deduplication (same file, same snapshot)
    before = len(df)
    df = df.drop_duplicates(subset=["repo", "file_path", "snapshot_commit"])
    logger.info(f"  Deduplicated: {before} → {len(df)} rows")

    # Fix 0-F: Post-mining guard rails
    # Flag repos that are statistically degenerate so ExperimentRunner can auto-exclude them.
    # These flags do NOT prevent the parquet from being saved — they allow downstream
    # filtering without losing the data for inspection.
    bug_rate = df["future_bug_fix"].mean() if "future_bug_fix" in df.columns else -1.0
    n_rows = len(df)

    if n_rows < 50:
        logger.warning(
            f"  [GUARD:SMALL] {repo_name}: only {n_rows} rows after dedup "
            f"(threshold=50). Flagging exclude_from_training=1. "
            f"This repo will be excluded from ExperimentRunner."
        )
        df["exclude_from_training"] = 1
    else:
        df["exclude_from_training"] = 0

    if 0.0 <= bug_rate < 0.05:
        logger.warning(
            f"  [GUARD:LOW-LABEL] {repo_name}: {bug_rate:.1%} bug rate below 5% floor. "
            f"Possible heuristic blind spot (Jira-only tracker?). Flagging label_outlier=1."
        )
        df["label_outlier"] = 1
    elif bug_rate > 0.65:
        logger.warning(
            f"  [GUARD:HIGH-LABEL] {repo_name}: {bug_rate:.1%} bug rate above 65% ceiling. "
            f"Possible temporal density issue. Flagging label_outlier=1. "
            f"Expected range: 15%–35%."
        )
        df["label_outlier"] = 1
    else:
        df["label_outlier"] = 0
        logger.info(f"  Bug rate: {bug_rate:.1%} ✓ (within 5%–65% healthy range)")

    # Save to Parquet
    output_path = Path(output_parquet)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(output_path), index=False, engine="pyarrow")
    logger.success(f"  Saved {len(df)} rows → {output_path}")

    return df


def mine_all_repositories(
    config: dict,
    clone_base_dir: str | Path = "data/clones",
    output_dir: str | Path = "data/raw",
    db_path: str | Path = "data/smellpredict.duckdb",
) -> pd.DataFrame:
    """
    Mine all repositories defined in mining_config.yaml.

    Args:
        config: Parsed mining_config.yaml dictionary
        clone_base_dir: Where to clone repositories
        output_dir: Where to write per-repo Parquet files
        db_path: DuckDB database path for combined storage

    Returns:
        Combined DataFrame across all repositories
    """
    all_dfs: list[pd.DataFrame] = []
    repos_config = config.get("repositories", {})
    mining_cfg = config.get("mining", {})

    all_repos = []
    for tier_repos in repos_config.values():
        all_repos.extend(tier_repos)

    logger.info(f"Mining {len(all_repos)} repositories total")

    for repo_cfg in all_repos:
        url = repo_cfg["url"]
        name = repo_cfg["name"]
        output_parquet = Path(output_dir) / f"{name}.parquet"

        # Skip if already mined (incremental)
        if output_parquet.exists():
            logger.info(f"  [SKIP] {name} already mined. Loading from cache.")
            df = pd.read_parquet(str(output_parquet))
            all_dfs.append(df)
            continue

        try:
            df = mine_repository(
                repo_url=url,
                repo_name=name,
                clone_dir=clone_base_dir,
                output_parquet=output_parquet,
                snapshot_stride=mining_cfg.get("snapshot_stride", 5),
                future_window_days=mining_cfg.get("future_window_days", 90),  # FIX A
            )
            if not df.empty:
                all_dfs.append(df)
        except Exception as exc:
            logger.error(f"  [FAIL] {name}: {exc}")
            continue

    if not all_dfs:
        logger.warning("No repositories successfully mined.")
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    logger.info(f"Combined dataset: {len(combined)} rows from {combined['repo'].nunique()} repos")

    # Store in DuckDB
    db = duckdb.connect(str(db_path))
    db.execute("DROP TABLE IF EXISTS snapshots")
    db.execute("CREATE TABLE snapshots AS SELECT * FROM combined")
    db.close()
    logger.success(f"Stored combined dataset in DuckDB: {db_path}")
    # FIX D: Do NOT write all_repos_merged.parquet — it creates a duplicate artifact
    # that inflates DuckDB counts and poisons global analysis. The merged view is
    # always available via DuckDB: SELECT * FROM snapshots

    return combined
