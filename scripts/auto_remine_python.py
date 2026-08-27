"""
Python Re-Mine Auto-Launcher
============================
Polls Java mining completion, cleans stale Python data,
then starts the Python re-mine automatically.
Run ONCE and it will wait for Java to finish, then execute.
"""
import subprocess
import time
import sys
import os
from pathlib import Path

os.chdir(r"c:\Users\Asus\Downloads\New folder (3)\smellpredict")
sys.path.insert(0, "src")

JAVA_RAW_DIR = Path("data/java/raw")
PYTHON_RAW_DIR = Path("data/raw")
JAVA_TOTAL_REPOS = 50

def count_java_done():
    return len(list(JAVA_RAW_DIR.glob("*.parquet")))

print("=== Python Re-Mine Auto-Launcher ===")
print(f"Waiting for Java mining to complete ({JAVA_TOTAL_REPOS} repos)...")
print()

# Poll every 5 minutes until Java is done
while True:
    done = count_java_done()
    print(f"[{time.strftime('%H:%M:%S')}] Java progress: {done}/{JAVA_TOTAL_REPOS} repos complete", flush=True)
    if done >= JAVA_TOTAL_REPOS:
        print("Java mining COMPLETE. Starting Python cleanup and re-mine...")
        break
    time.sleep(300)  # check every 5 minutes

# Step 1: Delete stale Python parquets
print("\n--- Step 1: Cleaning stale Python data ---")
deleted = 0
for p in PYTHON_RAW_DIR.glob("*.parquet"):
    p.unlink()
    print(f"  Deleted: {p.name}")
    deleted += 1
print(f"  Deleted {deleted} stale parquet files.")

# Step 2: Clear Python DuckDB table
print("\n--- Step 2: Clearing Python DuckDB ---")
try:
    import duckdb
    db_path = "data/smellpredict.duckdb"
    if Path(db_path).exists():
        con = duckdb.connect(db_path)
        con.execute("DROP TABLE IF EXISTS snapshots")
        con.close()
        print(f"  Cleared snapshots table in {db_path}")
except Exception as e:
    print(f"  Warning: DuckDB clear failed: {e}")

# Step 3: Start Python mining
print("\n--- Step 3: Starting Python re-mine ---")
print(f"  Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("  Command: python main.py mine")
print()
result = subprocess.run(
    ["python", "main.py", "mine"],
    cwd=r"c:\Users\Asus\Downloads\New folder (3)\smellpredict"
)
print(f"\nPython mining finished with exit code: {result.returncode}")
