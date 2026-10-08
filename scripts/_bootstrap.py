"""Put <repo>/src on sys.path so scripts work without relying on the editable install.

(On macOS, Python 3.14 skips .pth files carrying the 'hidden' flag, which the OS re-applies inside .venv,
so `pip install -e .` alone is not reliable there.)
"""

import sys
from pathlib import Path

_SRC = str(Path(__file__).resolve().parent.parent / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
