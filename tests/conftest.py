import os
import sys

import numpy as np
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


@pytest.fixture
def rng():
    """Provide a seeded random number generator for reproducible tests."""
    return np.random.default_rng(12345)


@pytest.fixture
def csv_file(tmp_path):
    """Create a sample CSV file with numeric, categorical, and missing values."""
    p = tmp_path / "data.csv"
    p.write_text(
        "a,b,c,label\n"
        "1,2.5,x,0\n"
        "4,,y,1\n"
        "NA,6.0,x,1\n"
        "7,8.5,,0\n"
    )
    return p