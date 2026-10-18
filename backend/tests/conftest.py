import sys
from pathlib import Path

import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "slow: runs the model over the whole KDDTest+ file (can take a minute); "
        "skip with  -m \"not slow\"")


@pytest.fixture(scope="session")
def kddtest_frame():
    """The official KDDTest+ records (22,544 rows)."""
    path = REPO_ROOT / "data" / "KDDTest+.csv"
    if path.exists():
        return pd.read_csv(path)
    merged = REPO_ROOT / "data" / "NSL-KDD_merged.csv"
    frame = pd.read_csv(merged)
    return frame[frame["source_file"] == "test"].reset_index(drop=True)
