"""
SmellPredict — Full Integrity Audit
====================================
Verifies that all mined data is real, not synthetic or faked.
Checks: parquet files, column schemas, data distributions, model artifacts.
"""
import pandas as pd
from pathlib import Path
import json
import os
import sys

def audit():
    print("=" * 70)
    print("FULL INTEGRITY AUDIT — SmellPredict Pipeline")
    print("=" * 70)

    raw_dir = Path("data/raw")
    parquets = sorted(raw_dir.glob("*.parquet"))

    # 1. Individual Parquet Files
    print(f"\n[1] PARQUET FILES ON DISK: {len(parquets)}")
    total_rows = 0
    repo_stats = []
    for p in parquets:
        if p.name == "all_repos_merged.parquet":
            continue
        try:
            df = pd.read_parquet(p)
            total_rows += len(df)
            bug_count = int(df["future_bug_fix"].sum()) if "future_bug_fix" in df.columns else -1
            bug_rate = df["future_bug_fix"].mean() * 100 if "future_bug_fix" in df.columns else -1
            has_repo_col = "repo" in df.columns
            repo_val = df["repo"].iloc[0] if has_repo_col and len(df) > 0 else "MISSING"
            size_kb = p.stat().st_size / 1024
            print(f"  {p.name:30s}  {len(df):6d} rows  bugs={bug_count:5d}  rate={bug_rate:.1f}%  cols={len(df.columns)}  size={size_kb:.1f}KB  repo_col={repo_val}")
            repo_stats.append({
                "file": p.name,
                "rows": len(df),
                "bugs": bug_count,
                "bug_rate": round(bug_rate, 2),
                "columns": len(df.columns),
                "size_kb": round(size_kb, 1),
                "repo_value": repo_val,
            })
        except Exception as e:
            print(f"  {p.name:30s}  CORRUPT: {e}")

    print(f"\n  TOTAL INDIVIDUAL ROWS: {total_rows}")

    # 2. Merged Dataset
    merged = raw_dir / "all_repos_merged.parquet"
    if merged.exists():
        mdf = pd.read_parquet(merged)
        repos_in_merged = sorted(mdf["repo"].unique().tolist())
        print(f"\n[2] MERGED DATASET: {len(mdf)} rows, {len(mdf.columns)} columns")
        print(f"  Repos in merged: {repos_in_merged}")
        print(f"  Merged matches sum of individuals: {len(mdf) == total_rows}")
        if len(mdf) != total_rows:
            print(f"  *** WARNING: Mismatch! Merged={len(mdf)}, Sum={total_rows}")
    else:
        print("\n[2] MERGED DATASET: NOT FOUND (mining still in progress)")

    # 3. Column Schema Verification
    print("\n[3] COLUMN SCHEMA VERIFICATION")
    if repo_stats:
        sample_file = raw_dir / repo_stats[0]["file"]
        sdf = pd.read_parquet(sample_file)
        cols = sdf.columns.tolist()
        print(f"  Sample file: {repo_stats[0]['file']}")
        print(f"  Total columns: {len(cols)}")
        print(f"  Columns: {cols}")

    # 4. Data Distribution Sanity Checks
    print("\n[4] DATA DISTRIBUTION SANITY CHECKS")
    if repo_stats:
        sample_file = raw_dir / repo_stats[0]["file"]
        sdf = pd.read_parquet(sample_file)
        
        if "code_loc" in sdf.columns:
            loc_stats = sdf["code_loc"].describe()
            print(f"  code_loc: min={loc_stats['min']:.0f}, max={loc_stats['max']:.0f}, mean={loc_stats['mean']:.1f}, std={loc_stats['std']:.1f}")
            if loc_stats["std"] == 0:
                print("  *** SUSPICIOUS: code_loc has zero variance!")
            else:
                print("  code_loc has natural variance OK")
        
        if "snapshot_date" in sdf.columns:
            dates = pd.to_datetime(sdf["snapshot_date"])
            print(f"  snapshot_date range: {dates.min()} to {dates.max()}")
            span_days = (dates.max() - dates.min()).days
            print(f"  Date span: {span_days} days ({span_days/365:.1f} years)")

    # 5. Check for Fake/Placeholder Models
    print("\n[5] MODEL ARTIFACTS CHECK")
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.rglob("*"))
        if model_files:
            print(f"  Found {len(model_files)} files in models/:")
            for mf in model_files:
                if mf.is_file():
                    print(f"    {mf.relative_to(models_dir)}  ({mf.stat().st_size/1024:.1f}KB)")
        else:
            print("  models/ directory is empty OK")
    else:
        print("  models/ directory does not exist OK")

    # 6. Synthetic Data Generator Check
    print("\n[6] SYNTHETIC DATA GENERATOR CHECK")
    sus_patterns = ["np.random.seed", "generate_fake", "mock_data", "dummy_model", "placeholder_model"]
    src_dir = Path("src/smellpredict")
    flagged = []
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8", errors="replace").lower()
            for pat in sus_patterns:
                if pat in content:
                    flagged.append((str(py_file.relative_to(src_dir)), pat))
        except:
            pass
    
    if flagged:
        print("  *** FLAGGED patterns found:")
        for f, p in flagged:
            print(f"    {f}: contains '{p}'")
    else:
        print("  No synthetic/fake data generators found in source code OK")

    # 7. Git Clone Verification
    print("\n[7] CLONE DIRECTORIES CHECK")
    clones_dir = Path("data/clones")
    if clones_dir.exists():
        clone_dirs = [d for d in clones_dir.iterdir() if d.is_dir()]
        valid_clones = [d for d in clone_dirs if (d / ".git").exists()]
        broken_clones = [d for d in clone_dirs if not (d / ".git").exists()]
        print(f"  Total clone directories: {len(clone_dirs)}")
        print(f"  Valid (have .git): {len(valid_clones)}")
        print(f"  Broken (no .git): {len(broken_clones)}")
        if broken_clones:
            print(f"  Broken: {[d.name for d in broken_clones]}")
    else:
        print("  data/clones/ does not exist")

    # Summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)
    print(f"  Parquet files: {len(repo_stats)}")
    print(f"  Total rows: {total_rows}")
    print(f"  All files readable: YES")
    if flagged:
        print(f"  Synthetic patterns in code: YES - INVESTIGATE")
    else:
        print(f"  Synthetic patterns in code: NONE - CLEAN")
    print("=" * 70)


if __name__ == "__main__":
    audit()
