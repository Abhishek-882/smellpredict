"""
SmellPredict — Java Repository Mining Module
=============================================
Mines Java repositories using PyDriller to extract:

  - File snapshots at fixed commit strides (every 5th commit)
  - Java source-code metrics and smell features per snapshot
  - Development-history metrics from git commit history
  - Bug-fix labels using the language-agnostic heuristic

Architecture mirrors miner.py exactly — same functions, same DuckDB schema,
but filtered to .java files and using java_extractor instead of extractor.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from math import log2
from pathlib import Path
from typing import Iterator, Optional

import duckdb
import pandas as pd
import yaml
from loguru import logger
from tqdm import tqdm

try:
    from pydriller import Repository
    HAS_PYDRILLER = True
except ImportError:
    HAS_PYDRILLER = False
    logger.warning("PyDriller not installed. Mining features disabled.")

# ─────────────────────────────────────────────────────────────────────────────
# Dedicated Java log file
# ─────────────────────────────────────────────────────────────────────────────

log_dir = Path("logs")
log_dir.mkdir(parents=True, exist_ok=True)
logger.add(
    log_dir / "java_mining.log",
    rotation="50 MB",
    retention="10 days",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} - {message}",
    encoding="utf-8",
    filter=lambda record: True,   # capture all from java_miner
)

from smellpredict.features.java_extractor import extract_java_metrics, extract_java_smells
from smellpredict.labeling.heuristic import score_commit


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

JAVA_EXCLUDE_PATTERNS = [
    re.compile(r'Test\w+\.java$'),
    re.compile(r'\w+Tests?\.java$'),
    re.compile(r'\w+IT\.java$'),
    re.compile(r'\w+ITCase\.java$'),
    re.compile(r'.*/generated(-sources)?/.*\.java$'),
    re.compile(r'.*/protobuf/.*\.java$'),
    re.compile(r'.*/target/.*\.java$'),
    re.compile(r'.*/build/.*\.java$'),
    re.compile(r'.*/test[s]?/.*\.java$'),
    re.compile(r'package-info\.java$'),
    re.compile(r'module-info\.java$'),
]

MAX_FILE_SIZE_BYTES = 500 * 1024   # 500 KB

DUCKDB_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS java_snapshots (
    snapshot_id                    VARCHAR PRIMARY KEY,
    repo                           VARCHAR NOT NULL,
    file_path                      VARCHAR NOT NULL,
    file_id                        VARCHAR NOT NULL,
    commit_hash                    VARCHAR NOT NULL,
    commit_date                    TIMESTAMP NOT NULL,
    future_bug_fix                 INTEGER NOT NULL,
    parse_fallback                 BOOLEAN DEFAULT FALSE,
    -- Code Metrics (Group A)
    code_loc                       INTEGER,
    code_sloc                      INTEGER,
    code_blank_lines               INTEGER,
    code_comment_lines             INTEGER,
    code_comment_density           DOUBLE,
    code_function_count            INTEGER,
    code_class_count               INTEGER,
    code_import_count              INTEGER,
    code_avg_function_size         DOUBLE,
    code_max_function_size         INTEGER,
    code_avg_param_count           DOUBLE,
    code_max_param_count           INTEGER,
    code_max_nesting_depth         INTEGER,
    code_avg_cyclomatic_complexity DOUBLE,
    code_max_cyclomatic_complexity INTEGER,
    code_halstead_volume           DOUBLE,
    code_halstead_difficulty       DOUBLE,
    code_halstead_effort           DOUBLE,
    code_halstead_bugs             DOUBLE,
    code_maintainability_index     DOUBLE,
    code_cognitive_complexity      INTEGER,
    -- Smell Features (Group B)
    has_long_method                INTEGER,
    has_long_param_list            INTEGER,
    has_large_class                INTEGER,
    has_deep_nesting               INTEGER,
    has_high_complexity            INTEGER,
    long_method_count              INTEGER,
    long_param_count               INTEGER,
    large_class_count              INTEGER,
    deep_nesting_count             INTEGER,
    high_complexity_count          INTEGER,
    total_smells                   INTEGER,
    -- History Metrics (Group C)
    previous_file_commits          INTEGER,
    previous_bug_fixes             INTEGER,
    contributors                   INTEGER,
    recent_file_commits            INTEGER,
    code_churn_history             INTEGER,
    file_age_days                  DOUBLE,
    days_since_last_change         DOUBLE,
    developer_experience           DOUBLE,
    ownership_concentration        DOUBLE,
    commit_message_entropy         DOUBLE,
    avg_commit_size                DOUBLE,
    avg_time_between_commits       DOUBLE,
    has_multiple_contributors      INTEGER,
    is_recently_touched            INTEGER
)
"""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (mirrors miner.py)
# ─────────────────────────────────────────────────────────────────────────────

def _should_exclude_java(file_path: str) -> bool:
    fp = file_path.replace("\\", "/")
    if not fp.endswith(".java"):
        return True
    return any(p.search(fp) for p in JAVA_EXCLUDE_PATTERNS)


def canonical_file_id(repo_name: str, file_path: str) -> str:
    key = f"{repo_name}::{file_path.replace(os.sep, '/')}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _force_rmtree(path: Path | str):
    """Robustly remove directory tree on Windows handling read-only file locks."""
    p_path = Path(path)
    if not p_path.exists():
        return
    import shutil
    import stat

    def _on_error(func, p, exc_info):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except Exception:
            pass

    try:
        shutil.rmtree(p_path, onerror=_on_error)
    except Exception as e:
        logger.debug(f"Failed to remove {p_path}: {e}")


def clone_repo(url: str, target_dir: str | Path, max_retries: int = 3) -> Path:
    target = Path(target_dir)
    target.parent.mkdir(parents=True, exist_ok=True)

    # If valid git repo exists, reuse it
    if (target / ".git").exists():
        check = subprocess.run(
            ["git", "-C", str(target), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True
        )
        if check.returncode == 0 and check.stdout.strip() == "true":
            logger.info(f"Reusing existing valid clone: {target}")
            return target
        else:
            logger.warning(f"Existing clone at {target} is broken — wiping and re-cloning")
            _force_rmtree(target)

    for attempt in range(1, max_retries + 1):
        logger.info(f"Cloning {url} → {target} (Attempt {attempt}/{max_retries})")
        if target.exists():
            _force_rmtree(target)

        result = subprocess.run(
            ["git", "-c", "core.longpaths=true", "clone", url, str(target)],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            logger.success(f"Cloned {url}")
            return target
        logger.warning(f"Clone attempt {attempt} failed for {url}: {result.stderr.strip()}")
        time.sleep(3)
    raise RuntimeError(f"git clone failed after {max_retries} attempts: {url}")


def compute_history_metrics(
    file_path: str,
    repo_path: str | Path,
    snapshot_date: datetime,
    window_days_recent: int = 30,
    window_days_active: int = 7,
) -> dict:
    """Compute development-history metrics from git log (language-agnostic)."""
    h = {
        "previous_file_commits": 0, "previous_bug_fixes": 0, "contributors": 0,
        "recent_file_commits": 0, "code_churn_history": 0, "file_age_days": 0.0,
        "days_since_last_change": 0.0, "developer_experience": 0.0,
        "ownership_concentration": 1.0, "commit_message_entropy": 0.0,
        "avg_commit_size": 0.0, "avg_time_between_commits": 0.0,
        "has_multiple_contributors": 0, "is_recently_touched": 0,
    }
    try:
        cutoff = snapshot_date.strftime("%Y-%m-%dT%H:%M:%S")
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log", "-n", "100",
             f"--before={cutoff}", "--format=%H|%ae|%ad|%s",
             "--date=iso-strict", "--", file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=10
        )
        if result.returncode != 0 or not result.stdout.strip():
            return h

        lines = [l for l in result.stdout.strip().split("\n") if "|" in l]
        if not lines:
            return h

        commits, authors, dates, messages = [], [], [], []
        for line in lines:
            parts = line.split("|", 3)
            if len(parts) < 4:
                continue
            commits.append(parts[0])
            authors.append(parts[1])
            try:
                dates.append(datetime.fromisoformat(parts[2].strip()))
            except Exception:
                pass
            messages.append(parts[3])

        if not commits:
            return h

        now = snapshot_date.replace(tzinfo=None) if snapshot_date.tzinfo else snapshot_date
        date_objs = []
        for d in dates:
            d_naive = d.replace(tzinfo=None) if d.tzinfo else d
            date_objs.append(d_naive)

        h["previous_file_commits"] = len(commits)
        h["contributors"] = len(set(authors))
        h["has_multiple_contributors"] = int(len(set(authors)) > 1)

        # Bug fix count
        from smellpredict.labeling.heuristic import score_commit as _score_commit
        h["previous_bug_fixes"] = sum(1 for m in messages if _score_commit(m).is_bug_fix)

        # Recent commits
        recent_cutoff = now - __import__('datetime').timedelta(days=window_days_recent)
        h["recent_file_commits"] = sum(1 for d in date_objs if d >= recent_cutoff)

        # Activity
        active_cutoff = now - __import__('datetime').timedelta(days=window_days_active)
        h["is_recently_touched"] = int(any(d >= active_cutoff for d in date_objs))

        # File age
        if date_objs:
            oldest = min(date_objs)
            newest = max(date_objs)
            h["file_age_days"] = round((now - oldest).days, 1)
            h["days_since_last_change"] = round((now - newest).days, 1)

        # Ownership concentration (Herfindahl index)
        author_counts = defaultdict(int)
        for a in authors:
            author_counts[a] += 1
        total = max(len(authors), 1)
        shares = [c / total for c in author_counts.values()]
        h["ownership_concentration"] = round(sum(s ** 2 for s in shares), 4)

        # Commit message entropy
        msg_words = " ".join(messages).lower().split()
        if msg_words:
            word_freq = defaultdict(int)
            for w in msg_words:
                word_freq[w] += 1
            n = len(msg_words)
            h["commit_message_entropy"] = round(
                -sum((c / n) * log2(c / n) for c in word_freq.values()), 4
            )

        # Inter-commit time
        if len(date_objs) >= 2:
            sorted_dates = sorted(date_objs)
            gaps = [(sorted_dates[i + 1] - sorted_dates[i]).total_seconds() / 3600
                    for i in range(len(sorted_dates) - 1)]
            h["avg_time_between_commits"] = round(sum(gaps) / len(gaps), 2)

        # Developer experience
        h["developer_experience"] = round(min(len(commits) / 50.0, 1.0), 4)

    except Exception as e:
        logger.debug(f"History metrics failed for {file_path}: {e}")

    return h


def _compute_future_label(
    file_path: str,
    repo_path: str | Path,
    snapshot_date: datetime,
    window_days: int = 90,
) -> int:
    """Return 1 if this file has a bug-fix commit within window_days after snapshot_date.

    Fix (1): Normalize Windows backslashes to forward slashes before passing
    to git log. On Windows, subprocess passes backslash paths to git which
    silently treats them as non-matching path filters, causing all labels = 0.
    """
    try:
        after = snapshot_date.strftime("%Y-%m-%dT%H:%M:%S")
        cutoff_date = snapshot_date + timedelta(days=window_days)   # Fix 0-B: direct import, not __import__()
        before = cutoff_date.strftime("%Y-%m-%dT%H:%M:%S")

        # BUG FIX (1): git log path filter requires forward slashes on all platforms.
        # Windows pydriller returns mod.new_path with backslashes; normalize here.
        git_file_path = file_path.replace("\\", "/")

        result = subprocess.run(
            ["git", "-C", str(repo_path), "log",
             f"--after={after}", f"--before={before}",
             "--format=%s", "--", git_file_path],
            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=15
        )
        if result.returncode != 0 or not result.stdout.strip():
            return 0

        from smellpredict.labeling.heuristic import score_commit as _score_commit
        for msg in result.stdout.strip().split("\n"):
            if msg.strip() and _score_commit(msg.strip()).is_bug_fix:
                return 1
    except Exception:
        pass
    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Core Mining Functions
# ─────────────────────────────────────────────────────────────────────────────

def mine_java_repository(
    repo_url: str,
    repo_name: str,
    clone_dir: Path,
    output_dir: Path,
    db_path: Path,
    commit_stride: int = 5,
    max_commits: int = 500,
    snapshot_window_days: int = 90,
) -> pd.DataFrame:
    """
    Mine a single Java repository and save features to DuckDB + parquet.

    Returns DataFrame of extracted snapshots.
    """
    if not HAS_PYDRILLER:
        raise RuntimeError("PyDriller is required for mining. pip install pydriller")

    repo_clone_path = clone_dir / repo_name

    # Clone if not already present
    if not (repo_clone_path / ".git").exists():
        clone_repo(repo_url, repo_clone_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{repo_name}.parquet"

    # Skip if already fully mined
    if parquet_path.exists():
        logger.info(f"Skipping {repo_name} — already mined at {parquet_path}")
        return pd.read_parquet(parquet_path)

    logger.info(f"Mining Java repository: {repo_name}")

    # Collect commits
    try:
        all_commits = list(Repository(str(repo_clone_path)).traverse_commits())
    except Exception as e:
        logger.error(f"Failed to traverse commits for {repo_name}: {e}")
        return pd.DataFrame()

    # BUG FIX (2): Previously took the first max_commits commits (oldest history).
    # For large repos like Elasticsearch, those early commits (2010-2011) reference
    # file paths that were later renamed/deleted, causing _compute_future_label to
    # return 0 for every snapshot. Fix: sample evenly across the FULL commit history
    # so we always include recent, stable-path commits where bug-fix labels work.
    if len(all_commits) <= max_commits:
        sampled = all_commits
    else:
        indices = [int(i * (len(all_commits) - 1) / (max_commits - 1)) for i in range(max_commits)]
        sampled = [all_commits[i] for i in indices]
    logger.info(f"{repo_name}: {len(all_commits)} total commits → {len(sampled)} sampled (evenly distributed)")

    rows = []
    seen_ids: set[str] = set()

    for c_idx, commit in enumerate(tqdm(sampled, desc=f"Java-Snapshots [{repo_name}]", unit="commit"), 1):
        commit_date = commit.committer_date
        if commit_date.tzinfo:
            commit_date = commit_date.replace(tzinfo=None)

        # Heartbeat status dump for live monitor
        try:
            status_file = Path("data/java/mining_status.json")
            status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(status_file, "w", encoding="utf-8") as sf:
                json.dump({
                    "active_repo": repo_name,
                    "active_current": c_idx,
                    "active_total": len(sampled),
                    "active_pct": round((c_idx / max(len(sampled), 1)) * 100, 1),
                    "snapshots_count": len(rows),
                    "timestamp": time.time(),
                }, sf)
        except Exception:
            pass

        for mod in commit.modified_files:
            if mod.filename is None or not mod.filename.endswith(".java"):
                continue
            file_path = mod.new_path or mod.old_path or mod.filename
            if file_path is None or _should_exclude_java(file_path):
                continue

            source = mod.source_code
            if not source or len(source.encode()) > MAX_FILE_SIZE_BYTES:
                continue

            # Unique snapshot ID
            snap_id = hashlib.sha256(
                f"{repo_name}::{file_path}::{commit.hash}".encode()
            ).hexdigest()[:24]
            if snap_id in seen_ids:
                continue
            seen_ids.add(snap_id)

            # Extract features
            try:
                metrics = extract_java_metrics(source)
                smells  = extract_java_smells(source, metrics)
                history = compute_history_metrics(file_path, repo_clone_path, commit_date)
                label   = _compute_future_label(file_path, repo_clone_path, commit_date, snapshot_window_days)
            except Exception as e:
                logger.debug(f"Feature extraction failed for {file_path}@{commit.hash[:8]}: {e}")
                continue

            file_id = canonical_file_id(repo_name, file_path)

            row = {
                "snapshot_id":   snap_id,
                "repo":          repo_name,
                "file_path":     file_path,
                "file_id":       file_id,
                "commit_hash":   commit.hash,
                "commit_date":   commit_date,
                "future_bug_fix": label,
                "parse_fallback": metrics.get("parse_fallback", False),
                # Code Metrics
                "code_loc":                       metrics.get("loc", 0),
                "code_sloc":                      metrics.get("sloc", 0),
                "code_blank_lines":               metrics.get("blank_lines", 0),
                "code_comment_lines":             metrics.get("comment_lines", 0),
                "code_comment_density":           metrics.get("comment_density", 0.0),
                "code_function_count":            metrics.get("function_count", 0),
                "code_class_count":               metrics.get("class_count", 0),
                "code_import_count":              metrics.get("import_count", 0),
                "code_avg_function_size":         metrics.get("avg_function_size", 0.0),
                "code_max_function_size":         metrics.get("max_function_size", 0),
                "code_avg_param_count":           metrics.get("avg_param_count", 0.0),
                "code_max_param_count":           metrics.get("max_param_count", 0),
                "code_max_nesting_depth":         metrics.get("max_nesting_depth", 0),
                "code_avg_cyclomatic_complexity": metrics.get("avg_cyclomatic_complexity", 0.0),
                "code_max_cyclomatic_complexity": metrics.get("max_cyclomatic_complexity", 0),
                "code_halstead_volume":           metrics.get("halstead_volume", 0.0),
                "code_halstead_difficulty":       metrics.get("halstead_difficulty", 0.0),
                "code_halstead_effort":           metrics.get("halstead_effort", 0.0),
                "code_halstead_bugs":             metrics.get("halstead_bugs", 0.0),
                "code_maintainability_index":     metrics.get("maintainability_index", 0.0),
                "code_cognitive_complexity":      metrics.get("cognitive_complexity", 0),
                # Smell Features
                **{k: smells.get(k, 0) for k in [
                    "has_long_method", "has_long_param_list", "has_large_class",
                    "has_deep_nesting", "has_high_complexity",
                    "long_method_count", "long_param_count", "large_class_count",
                    "deep_nesting_count", "high_complexity_count", "total_smells",
                ]},
                # History Metrics
                **history,
            }
            rows.append(row)

    if not rows:
        logger.warning(f"No Java snapshots extracted from {repo_name}")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Fix 0-G: Post-mining guard rails (mirrors miner.py Fix 0-F)
    # Flag repos whose label rate or row count is degenerate.
    # Flags are saved into the parquet and DuckDB so ExperimentRunner can filter them.
    n_rows = len(df)
    bug_rate = df["future_bug_fix"].mean() if "future_bug_fix" in df.columns else -1.0

    if n_rows < 50:
        logger.warning(
            f"  [GUARD:SMALL] {repo_name}: only {n_rows} rows "
            f"(threshold=50). Flagging exclude_from_training=1."
        )
        df["exclude_from_training"] = 1
    else:
        df["exclude_from_training"] = 0

    if 0.0 <= bug_rate < 0.05:
        logger.warning(
            f"  [GUARD:LOW-LABEL] {repo_name}: {bug_rate:.1%} bug rate below 5% floor. "
            f"Possible Jira-only tracker with no GitHub issue refs. Flagging label_outlier=1."
        )
        df["label_outlier"] = 1
    elif bug_rate > 0.65:
        logger.warning(
            f"  [GUARD:HIGH-LABEL] {repo_name}: {bug_rate:.1%} bug rate above 65% ceiling. "
            f"Temporal density issue suspected. Flagging label_outlier=1."
        )
        df["label_outlier"] = 1
    else:
        df["label_outlier"] = 0
        logger.info(f"  Bug rate: {bug_rate:.1%} ✓ (within healthy 5%–65% range)")

    # Persist to parquet
    df.to_parquet(parquet_path, index=False)
    logger.success(f"Saved {len(df)} snapshots for {repo_name} → {parquet_path}")

    # Upsert into DuckDB
    try:
        conn = duckdb.connect(str(db_path))
        conn.execute(DUCKDB_CREATE_TABLE)
        conn.execute("INSERT OR REPLACE INTO java_snapshots SELECT * FROM df")
        conn.close()
        logger.info(f"DuckDB updated: java_snapshots += {len(df)} rows from {repo_name}")
    except Exception as e:
        logger.error(f"DuckDB upsert failed for {repo_name}: {e}")

    return df


def mine_all_java_repositories(
    config: dict,
    clone_base_dir: Path,
    output_dir: Path,
    db_path: Path,
) -> pd.DataFrame:
    """
    Mine all Java repositories defined in config (tier1/tier2/tier3).
    Returns merged DataFrame of all snapshots.
    """
    mining_cfg = config.get("mining", {})
    commit_stride    = mining_cfg.get("commit_stride", 5)
    max_commits      = mining_cfg.get("max_commits_per_repo", 500)
    window_days      = mining_cfg.get("snapshot_window_days", 90)

    all_repos = []
    for tier_name, tier_repos in config.get("repositories", {}).items():
        for repo in tier_repos:
            all_repos.append((repo["url"], repo["name"], tier_name))

    logger.info(f"Starting Java mining: {len(all_repos)} repositories")
    all_dfs = []

    for repo_url, repo_name, tier in all_repos:
        try:
            df = mine_java_repository(
                repo_url=repo_url,
                repo_name=repo_name,
                clone_dir=clone_base_dir,
                output_dir=output_dir,
                db_path=db_path,
                commit_stride=commit_stride,
                max_commits=max_commits,
                snapshot_window_days=window_days,
            )
            if not df.empty:
                df["tier"] = tier
                all_dfs.append(df)
                logger.info(f"[{tier}] {repo_name}: {len(df)} snapshots")
        except Exception as e:
            logger.error(f"Failed to mine {repo_name}: {e}")

    if not all_dfs:
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True)

    # Save merged parquet
    merged_path = output_dir / "all_java_merged.parquet"
    merged.to_parquet(merged_path, index=False)
    logger.success(
        f"Java mining complete: {len(merged):,} snapshots from {len(all_dfs)} repos → {merged_path}"
    )
    return merged
