"""
FloodSense-AI: Comprehensive Sen1Floods11 Dataset Loader Test Suite
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import unittest
from test_sen1floods11_dataset import TestSen1Floods11Dataset

if __name__ == "__main__":
    unittest.main()

