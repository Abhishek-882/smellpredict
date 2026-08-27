import sys
from pathlib import Path

# Ensure src/ is on sys.path for test discovery without requiring pip install -e .
src_path = str(Path(__file__).parent.parent / "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)
