"""
Shared pytest configuration

Puts src/ on sys.path so tests import the project modules the same way
src/main.py does.
"""

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
