"""Put the flat src/ directory on sys.path so tests import modules the same way
the pipeline scripts do (scripts run with src/ as sys.path[0])."""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
