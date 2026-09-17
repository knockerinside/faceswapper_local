import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parents[1]
for p in [str(root_dir), str(root_dir / "realtime_faceswap")]:
    if p not in sys.path:
        sys.path.insert(0, p)

from realtime_faceswap.tools.diagnostics import print_status_table

if __name__ == "__main__":
    print_status_table()
